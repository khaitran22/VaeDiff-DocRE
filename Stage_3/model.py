import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import *
import torch.nn.functional as F
from axial_attention import AxialAttention, AxialImageTransformer
import numpy as np
import math
from itertools import accumulate
import copy

from dist import Normal

def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


class AxialTransformer_by_entity(nn.Module):
    def  __init__(self, emb_size = 768, dropout = 0.1, num_layers = 2, dim_index = -1, heads = 8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_attns = nn.ModuleList([AxialAttention(dim = self.emb_size, dim_index = dim_index, heads = heads, num_dimensions = num_dimensions, ) for i in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)] )
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)] )
    def forward(self, x):
        for idx in range(self.num_layers):
          x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
          x = self.ffns[idx](x)
          x = self.ffn_dropouts[idx](x)
          x = self.lns[idx](x)
        return x


class AxialEntityTransformer(nn.Module):
    def  __init__(self, emb_size = 768, dropout = 0.1, num_layers = 2, dim_index = -1, heads = 8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_img_transformer = AxialImageTransformer()
        self.axial_attns = nn.ModuleList([AxialAttention(dim = self.emb_size, dim_index = dim_index, heads = heads, num_dimensions = num_dimensions, ) for i in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)] )
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)] )
    def forward(self, x):
        for idx in range(self.num_layers):
          x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
          x = self.ffns[idx](x)
          x = self.ffn_dropouts[idx](x)
          x = self.lns[idx](x)
        return x


class DocREModel_KD(nn.Module):
    def __init__(self, args, config, model, emb_size=1024, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.emb_size = emb_size
        self.block_size = block_size
        
        self.loss_fnt = PMTEMloss(args.lambda_1, args.lambda_2)
        self.scl_fnt = SCLoss(args.tau, args.tau_base)

        self.args = args
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.entity_criterion = nn.CrossEntropyLoss()
        self.bin_criterion = nn.CrossEntropyLoss()
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.projection = nn.Linear(emb_size * block_size, config.hidden_size, bias=False)
        self.classifier = nn.Linear(config.hidden_size , config.num_labels)
        self.mse_criterion = nn.MSELoss()
        self.axial_transformer = AxialTransformer_by_entity(emb_size = config.hidden_size, dropout=0.0, num_layers=6, heads=8)
        self.emb_size = emb_size
        self.threshold = nn.Threshold(0,0)
        self.block_size = block_size
        self.num_labels = num_labels
    
    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        rss_2 = []
        sent_embs = []
        batch_entity_embs = []
        b, seq_l, h_size = sequence_output.size()
        n_e = 42
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            entity_lens = []

            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                     

                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            
            pad_hs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_ts = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, h_size)
            pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, h_size)


            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            m = torch.nn.Threshold(0,0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
            
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            pad_rs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, h_size)
            hss.append(pad_hs)
            tss.append(pad_ts)
            rss.append(pad_rs)
            batch_entity_embs.append(entity_embs)
        hss = torch.stack(hss, dim=0)
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)
        batch_entity_embs = torch.cat(batch_entity_embs, dim=0)
        return hss, rss, tss, batch_entity_embs

    def get_hrt_by_segment(self, sequence_output, attention, entity_pos, hts, segment_span):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        sent_embs = []
        batch_entity_embs = []
        #print(sequence_output.size(), attention.size())
        segment_start, segment_end = segment_span
        seg_start_idx = 0
        b, seq_l, h_size = sequence_output.size()
        n_e = max([len(x) for x in entity_pos])
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            entity_lens = []
            mask = []
            logit_mask = torch.zeros((n_e, n_e))
            for e_pos in entity_pos[i]:
                e_pos = [x for x in e_pos if (x[0] >= segment_start)  and (x[1] < segment_end)]
                
                if len(e_pos) > 1:
                    e_emb, e_att = [], []
                    exist = 1 
                    for start, end in e_pos:
                        start = start - segment_start
                        end = start - segment_start
                        if   start + offset < c :
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                elif len(e_pos) == 1:
                    start, end = e_pos[0]
                    start = start - segment_start
                    end = start - segment_start
                    exist = 1 
                    if  start + offset < c :
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                elif len(e_pos) == 0:
                    exist = 0
                    e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                mask.append(exist)
               
            for i_e in range(n_e):
                for j_e in range(n_e):
                    if mask[i_e]==1 and mask[j_e]==1:
                        logit_mask[i_e, j_e] = 1
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]

            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[0]).to(sequence_output.device)
                        
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0]).view(s_ne, s_ne, h_size)
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1]).view(s_ne, s_ne, h_size)
            
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            m = torch.nn.Threshold(0,0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
            rs = contract("ld,rl->rd", sequence_output[0], ht_att).view(s_ne, s_ne, h_size)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.stack(hss, dim=0)
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)
        return hss, rss, tss, logit_mask

    def encode_by_segment(self, input_ids, attention_mask, sentid_mask, ctx_window, stride):
        bsz, seq_len = input_ids.size()
        if seq_len <= ctx_window:
            segment_output, segment_attn = self.encode(input_ids,  attention_mask)
            return segment_output, segment_attn, [(0, seq_len)]
        else:
            segments = math.ceil((seq_len - ctx_window)/stride)
            batch_input_ids = []
            batch_input_attn = []
            segment_spans = []
            context_sz = 100
            max_len = stride * segments + ctx_window
            segment_input = torch.zeros((max_len)).to(input_ids)
            segment_attention = torch.zeros((max_len)).to(input_ids)
            segment_input[:seq_len] =  input_ids.squeeze(0)
            segment_attention[:seq_len] =  attention_mask.squeeze(0)
            for i in range(segments + 1):
                batch_input_ids.append(segment_input[ i * stride: i * stride + ctx_window])
                batch_input_attn.append(segment_attention[ i * stride: i * stride + ctx_window])
                segment_spans.append((i * stride, i * stride + ctx_window))
            batch_input_ids = torch.stack(batch_input_ids ,dim=0)
            batch_input_attn = torch.stack(batch_input_attn, dim=0)
            segment_output, segment_attn = self.encode(batch_input_ids,  batch_input_attn)
            return segment_output, segment_attn, segment_spans
 
    def encode_by_sentence(self, input_ids, attention_mask, sentid_mask, ctx_window, stride):
        bsz, seq_len = input_ids.size()
        
        segments = math.ceil((seq_len - ctx_window)/stride)        
        max_len = ctx_window * segments
        segment_input = torch.zeros((max_len)).to(input_ids)
        segment_attention = torch.zeros((max_len)).to(input_ids)
        segment_input[:seq_len] =  input_ids.squeeze(0)
        segment_attention[:seq_len] =  attention_mask.squeeze(0)
        segment_input = segment_input.view(segments, ctx_window).long()
        segment_attention = segment_attention.view(segments, ctx_window).long()
        segment_output, segment_attn = self.encode(segment_input,  segment_attention)
        return segment_output, segment_attn
    
    def get_logits_by_segment(self, segment_span, sequence_output, attention, entity_pos, hts):
        seg_start, seg_end = segment_span
        sequence_output = sequence_output.unsqueeze(0)
        attention = attention.unsqueeze(0)
        bs, seq_len, h_size = sequence_output.size()
        ne = len(entity_pos[0])
        hs_e, rs_e, ts_e, logit_mask = self.get_hrt_by_segment(sequence_output, attention, entity_pos, hts, segment_span)
        logit_mask = torch.tensor(logit_mask).clone().to(sequence_output).detach()

        hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e], dim=3)))        
        ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e], dim=3)))   
        b1_e = hs_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        b2_e = ts_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(bs, ne, ne, self.emb_size * self.block_size)
        
        feature = self.projection(bl_e) 
        feature = self.axial_transformer(feature) + feature
        logits = self.classifier(feature).squeeze()
        self_mask = (1 - torch.diag(torch.ones((ne)))).unsqueeze(-1).to(sequence_output)
        logits = logits * logit_mask.unsqueeze(-1)
        logits = logits * self_mask

        return logits, logit_mask

    def pos_augmenting_data(self, vae, ema_diffusion, diffusion, labels):
        model_kwargs = {
            'class_id': labels,
            'training': False,
            'w': self.args.w
        }

        bs, _ = labels.size()
        generated_shape = (bs, self.args.latent_size)
        sample_fn = (
            diffusion.p_sample_loop if not self.args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            ema_diffusion.ema_model,
            generated_shape,
            clip_denoised=self.args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        logits = vae.decode(sample)
        output, _ = logits.chunk(2, dim=1)

        return output
    
    def neg_augmenting_data(self, neg_feature):
        """
        Adding gaussian noise to the embedding
        """
        output = neg_feature + torch.rand_like(neg_feature)
        return output

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                negative_mask = None,
                augment_models=None,
                epoch=None
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        bs, seq_len, h_size = sequence_output.size()
        bs, num_heads, seq_len, seq_len = attention.size()
        ctx_window = 300
        stride = 25
        device = sequence_output.device.index
        ne = 42
        nes = [len(x) for x in entity_pos]
        hs_e, rs_e, ts_e, batch_entity_embs = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs_e_1 = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e], dim=3)))
        ts_e_1 = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e], dim=3)))
        b1_e = hs_e_1.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        b2_e = ts_e_1.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(bs, ne, ne, self.emb_size * self.block_size)
        
        if negative_mask is not None:
            bl_e = bl_e * negative_mask.unsqueeze(-1)

        feature =  self.projection(bl_e)
        feature = self.axial_transformer(feature) + feature

        label_embeddings = []
        logits_c = self.classifier(feature)
        self_mask = (1 - torch.diag(torch.ones((ne)))).unsqueeze(0).unsqueeze(-1).to(sequence_output)
        logits_classifier = logits_c * self_mask
        logits_classifier = torch.cat([logits_classifier.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])

        # inference + testing
        if labels is None:
            logits = logits_classifier.view(-1, self.config.num_labels)
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels), logits)
        
        # training
        else:
            if epoch >= self.args.warmup_epochs:
                self_mask_2d = torch.cat([self_mask.clone()[0, :nes[x], :nes[x] , :].reshape(-1, 1) for x in range(len(nes))]).bool().squeeze(1)
                feature_scl = torch.cat([feature.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.hidden_size) for x in range(len(nes))])
                feature_scl = feature_scl[self_mask_2d, :]

                # augmentation
                vae, ema_diffusion, diffusion = augment_models
                quant_each_augment = 2
                with torch.no_grad():
                    pos_aug_features = []
                    pos_aug_labels = []

                    # positive entity pairs
                    for i in range(len(labels)):
                        currr_labels = torch.tensor(labels[i]).to(device)
                        curr_pos_labels = currr_labels[currr_labels[:, 0] != 1, :]
                        curr_pos_labels = curr_pos_labels.repeat(quant_each_augment, 1)
                        pos_aug_feature = self.pos_augmenting_data(vae, ema_diffusion, diffusion, curr_pos_labels)

                        pos_aug_features.append(pos_aug_feature)
                        pos_aug_labels.append(curr_pos_labels)

                    aug_features = torch.vstack(pos_aug_features)
                    aug_labels = torch.vstack(pos_aug_labels)
                    
                    excl_self_labels = [torch.tensor(label) for label in labels]
                    excl_self_labels = torch.cat(excl_self_labels, dim=0).to(device)
                    excl_self_labels = excl_self_labels[self_mask_2d, :]

                feats_scl = torch.cat((feature_scl, aug_features), dim=0)
                aug_logits_classifier = self.classifier(aug_features)

            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(device)
            loss_classifier = self.loss_fnt(logits_classifier.view(-1, self.config.num_labels).float(), labels.float())
            output = loss_classifier
            
            if epoch >= self.args.warmup_epochs:
                loss_classifier_2 = self.loss_fnt(aug_logits_classifier.view(-1, self.config.num_labels).float(), aug_labels.float())

                # contrastive
                labels_scl = torch.cat((excl_self_labels, aug_labels), dim=0)
                idx_select = (labels_scl[:, 0] != 1).nonzero().squeeze(1).to(feats_scl.device)
                feats_scl = torch.index_select(feats_scl, 0, idx_select)
                labels_scl = torch.index_select(labels_scl, 0, idx_select)
                scl_loss = self.scl_fnt(feats_scl, labels_scl)
                output =  output + loss_classifier_2 + scl_loss
        
        return output


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma


class VAE(nn.Module):
    def __init__(self, args, config):
        super(VAE, self).__init__()

        self.args = args
        self.dim_h = config.hidden_size
        self.dim_emb = config.hidden_size // 4
        self.dim_z = args.latent_size
        
        ### encoder
        self.encoder = nn.Sequential(
            RMSNorm(self.dim_h),
            nn.Linear(self.dim_h, self.dim_emb),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim_emb, self.dim_emb),
            RMSNorm(self.dim_emb),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim_emb, self.dim_z),
        )
        self.mu = nn.Linear(self.dim_z, self.dim_z)
        self.log_sigma = nn.Linear(self.dim_z, self.dim_z)

        ### decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.dim_z, self.dim_emb),
            nn.LeakyReLU(0.2, inplace=True),
            RMSNorm(self.dim_emb),
            nn.Linear(self.dim_emb, self.dim_emb),
            nn.LeakyReLU(0.2, inplace=True),
            RMSNorm(self.dim_emb),
            nn.Linear(self.dim_emb, self.dim_h * 2)
        )

    def encode(self, features):
        z = self.encoder(features)
        mu = self.mu(z)
        log_sigma = self.log_sigma(z)
        return mu, log_sigma
    
    def decode(self, z):
        logits = self.decoder(z)
        return logits
    
    def decoder_output(self, logits):
        mu, log_sigma = torch.chunk(logits, 2, dim=1)
        return Normal(mu, log_sigma)
    
    def reconstruction_loss(self, output, input):
        log_p_input = output.log_p(input)
        return (-1) * log_p_input.sum()
    
    def forward(self, features):
        p_mu, p_log_sigma = self.encode(features)
        p_dist = Normal(p_mu, p_log_sigma)
        all_p_latent, _ = p_dist.sample()
        
        all_log_p = p_dist.log_p(all_p_latent)
        logits = self.decode(all_p_latent)
        
        return logits, all_p_latent, all_log_p