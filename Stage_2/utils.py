import torch
import random
import numpy as np
import json
import os
import pandas as pd

from model import VAE
from baseline import DocREModel_KD
from diffusion_transformers import DiffusionTransformer
from ema_pytorch import EMA

from improved_diffusion.script_util import *
from improved_diffusion.resample import *

def get_label_input_ids(data_dir, tokenizer):
    rel2id = json.load(open(os.path.join(data_dir, 'meta/rel2id.json')))
    rel2id['P0'] = 0
    wikiprop_df = pd.read_csv(os.path.join(data_dir, 'wikidata-properties.csv'))
    prop2item = {}
    id2item = {}

    for idx in range(len(wikiprop_df)):
        prop_id = wikiprop_df.iloc[idx]['ID']
        if prop_id in rel2id:
            rel_name = wikiprop_df.iloc[idx]['label']
            rel_alias = wikiprop_df.iloc[idx]['aliases']
            rel_description = wikiprop_df.iloc[idx]['description']
            id2item[rel2id[prop_id]] = {'name': rel_name, 'description': rel_description, 'alias':rel_alias}


    label_features = []
    for idx in range(len(id2item)):
        #for idx in id2item:
        name_token = tokenizer.tokenize(id2item[idx]['name'])
        description_token = tokenizer.tokenize(id2item[idx]['description'])
        alias_token = tokenizer.tokenize(str(id2item[idx]['alias']))
        #all_tokens = [tokenizer.cls_token] + name_token + [tokenizer.sep_token] + \
        #              alias_token +[tokenizer.sep_token] + description_token + [tokenizer.sep_token]
        all_tokens = [tokenizer.cls_token] + name_token + [tokenizer.sep_token] + description_token + [tokenizer.sep_token]
        #print(all_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        #print(input_ids)
        label_attn_mask = [1] * len(input_ids)
        label_features.append({'input_ids': input_ids, 'attention_masks': label_attn_mask})

    return label_features

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    sentid_mask = [ torch.cat([f["sentid_mask"] , torch.zeros((max_len - len(f["sentid_mask"])))]) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    mention_pos = [f["mention_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    mention_hts = [f["mention_hts"] for f in batch]
    padded_mention = [f["padded_mention"] for f in batch]
    padded_mention_mask = [f["padded_mention_mask"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    sentid_mask = torch.stack(sentid_mask)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    #labels = [torch.tensor(label) for label in labels]
    output = (input_ids, input_mask, labels, entity_pos, hts, mention_pos, mention_hts, padded_mention, padded_mention_mask, sentid_mask)
    return output

def label_collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long) 
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask)
    return output

def collate_fn_kd(batch):
    teacher_logits = None
    segment_spans = None
    entity_types = None
    max_len = max([len(f["input_ids"]) for f in batch])
    max_ent_len = max([len(f["entity_pos"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    sentid_mask = None
    neg_masks = None
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    mention_pos = [f["mention_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    mention_hts = [f["mention_hts"] for f in batch]
    padded_mention = [f["padded_mention"] for f in batch]
    padded_mention_mask = [f["padded_mention_mask"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    if "teacher_logits" in batch[0]:
        teacher_logits = [f["teacher_logits"] for f in batch]
    if "segment_span" in batch[0]:
        segment_spans = [f["segment_span"] for f in batch]
    if "entity_types" in batch[0]:
        entity_types = [x for f in batch for x in f['entity_types'] ]
        entity_types = [f['entity_types'] for f in batch ]
    if "negative_mask" in batch[0]:
        neg_masks = torch.stack([f['negative_mask'] for f in batch], dim=0)
    output = (input_ids, input_mask, labels, entity_pos, hts, mention_pos, mention_hts, padded_mention, padded_mention_mask, sentid_mask, teacher_logits, entity_types, segment_spans, neg_masks)
    return output

def proc_cond_info(labels, cond_info):
    indices = (labels == 1).nonzero(as_tuple=True)
    batch_indices, layer_indices = indices 
    selected_cond_info = cond_info[layer_indices]
    results = torch.zeros(labels.size(0), cond_info.size(1))

    for i in range(results.size(0)):
        batch_cond_infos = selected_cond_info[batch_indices == i]
        batch_cond_infos = batch_cond_infos.mean(dim=0, keepdim=True)
        results[i] = batch_cond_infos[0]

    return results

def kl_per_group(kl_all):
    kl_vals = torch.mean(kl_all, dim=0)
    kl_coeff_i = torch.abs(kl_all)
    kl_coeff_i = torch.mean(kl_coeff_i, dim=0, keepdim=True) + 0.01

    return kl_coeff_i, kl_vals

def kl_per_group_vada(all_log_q, all_neg_log_p):
    assert len(all_log_q) == len(all_neg_log_p)

    kl_all_list = []
    kl_diag = []
    for log_q, neg_log_p in zip(all_log_q, all_neg_log_p):
        kl_diag.append(torch.mean(torch.sum(neg_log_p + log_q, dim=-1), dim=0))
        kl_all_list.append(torch.sum(neg_log_p + log_q, dim=-1))

    kl_vals = torch.mean(torch.vstack(kl_all_list), dim=0)   # mean per group

    return kl_all_list, kl_vals, kl_diag

def kl_balancer(kl_all, kl_coeff=1.0, kl_balance=False, alpha_i=None):
    if kl_balance and kl_coeff < 1.0:
        alpha_i = alpha_i.unsqueeze(0)

        kl_all = torch.stack(kl_all, dim=1)
        kl_coeff_i, kl_vals = kl_per_group(kl_all)
        total_kl = torch.sum(kl_coeff_i)

        kl_coeff_i = kl_coeff_i / alpha_i * total_kl
        kl_coeff_i = kl_coeff_i / torch.mean(kl_coeff_i, dim=1, keepdim=True)
        kl = torch.sum(kl_all * kl_coeff_i.detach(), dim=1)

        # for reporting
        kl_coeffs = kl_coeff_i.squeeze(0)
    else:
        kl_all = torch.stack(kl_all, dim=0)
        kl_vals = torch.mean(kl_all, dim=0)
        kl = torch.sum(kl_all, dim=0)
        kl_coeffs = torch.ones(size=(len(kl_all),))

    return kl_coeff * kl, kl_coeffs, kl_vals

def build_models(args, config, encoder):
    # Baseline from Stage 1
    model = DocREModel_KD(args, config, encoder, num_labels=args.num_labels)
    model.load_state_dict(torch.load(args.load_baseline))
    model.to(args.device)
    
    # EP-VAE
    vae = VAE(args, config)
    vae.to(args.device)

    # Diff
    diffusion_net = DiffusionTransformer(
        latent_dim=args.latent_size,
        tx_dim=args.tx_dim,
        heads = args.tx_dim//64,
        tx_depth=args.tx_depth,
        class_conditional=args.class_conditional,
        self_condition=args.self_condition,
        scale_shift=args.scale_shift,
        num_classes=args.num_class-1
    )
    diffusion_net.to(args.device)

    # Diff EMA
    ema_diffusion = EMA(
        diffusion_net,
        beta=0.995,
        update_after_step=10,
        update_every=1
    )
    ema_diffusion.to(args.device)
    
    return model, vae, diffusion_net, ema_diffusion

def build_diff_process(args):
    # diffusion process
    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        sigma_small=args.sigma_small,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.timestep_respacing,
        train_prob_self_cond=args.train_prob_self_cond,
        factor=args.factor
    )

    # noise schedule
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    return diffusion, schedule_sampler