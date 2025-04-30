import argparse
import os
import math
import numpy as np
import torch

import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from model import DocREModel_KD
from utils import set_seed, collate_fn_kd, label_collate_fn, get_label_input_ids
from prepro import read_docred
from copy import deepcopy
from evaluation import *
from tqdm import tqdm

import autoRun
autoRun.choose_gpu(n_gpus=1, retry=True, min_gpu_memory=8000, sleep_time=30)


def get_lr(optimizer):
    lm_lr = optimizer.param_groups[0]['lr']
    classifier_lr = optimizer.param_groups[1]['lr']
    return lm_lr, classifier_lr


def train(args, model, train_features, dev_features, test_features, label_loader):
    def finetune(features, optimizer, num_epoch, num_steps, args):
        n_e = 42
        train_scores = []
        dev_scores = []
        best_score = -1
        tmp_features = get_random_mask(features, args.drop_prob)
        train_dataloader = DataLoader(tmp_features, batch_size=args.train_batch_size,
                                      shuffle=True, collate_fn=collate_fn_kd, drop_last=False)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch //
                          args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        grad_norms = []
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          'mention_pos': batch[5],
                          'mention_hts': batch[6],
                          'padded_mention': batch[7],
                          'padded_mention_mask': batch[8],
                          'sentid_mask': batch[9],
                          'teacher_logits': batch[10],
                          'entity_types': batch[11],
                          'segment_spans': batch[12],
                          'negative_mask': batch[13].to(args.device),
                          'label_loader': label_loader,
                          }

                loss = model(**inputs)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lm_lr, classifier_lr = get_lr(optimizer)
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1

                if (step + 1) == len(train_dataloader) - 1 or ((args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0) and num_steps > args.start_steps):
                    dev_score, dev_output = evaluate(
                        args, model, dev_features, label_loader, tag="dev")
                    lm_lr, classifier_lr = get_lr(optimizer)
                    print('Current Step: {:d},  Current LM lr: {:.5e}, Current Classifier lr: {:.5e}'.format(
                        num_steps, lm_lr, classifier_lr))

                    if dev_score > best_score:
                        best_score = dev_score
                        pred, logits = report(
                            args, model, test_features, label_loader)
                        test_score, test_output = evaluate(
                            args, model, test_features, label_loader, tag="test")
                        with open("./chkpoints/result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)

                    if args.save_last != "":
                        torch.save(model.state_dict(), args.save_last)

        return num_steps

    new_layer = ["extractor", "bilinear", "classifier",  "projection"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters(
        ) if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in new_layer)], "lr": args.classifier_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    num_steps = finetune(train_features, optimizer,
                         args.num_train_epochs, num_steps, args)


def evaluate(args, model, features, label_loader, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size,
                            shuffle=False, collate_fn=collate_fn_kd, drop_last=False)
    preds = []

    for batch in tqdm(dataloader, desc="Testing"):
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'mention_pos': batch[5],
                  'mention_hts': batch[6],
                  'padded_mention': batch[7],
                  'padded_mention_mask': batch[8],
                  'sentid_mask': batch[9],
                  'entity_types': batch[11],
                  'segment_spans': batch[12],
                  'label_loader': label_loader,
                  }
        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    best_f1, _, best_f1_ign, _, re_p, re_r, best_f1_freq, best_f1_long_tail, best_f1_intra, best_f1_inter, re_p_freq, re_r_freq, re_p_long_tail, re_r_long_tail = official_evaluate_benchmark(
        ans, args.data_dir, 'train_revised.json', f'{tag}_revised.json') if len(ans) > 0 else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    output = {
        tag + "_ign_F1": np.around(best_f1_ign * 100, decimals=2),
        tag + "_F1": np.around(best_f1 * 100, decimals=2),
        tag + "_p": np.around(re_p * 100, decimals=2),
        tag + "_r": np.around(re_r * 100, decimals=2),
        tag + "_F1_freq": np.around(best_f1_freq * 100, decimals=2),
        tag + "_p_freq": np.around(re_p_freq * 100, decimals=2),
        tag + "_r_freq": np.around(re_r_freq * 100, decimals=2),
        tag + "_F1_LT": np.around(best_f1_long_tail * 100, decimals=2),
        tag + "_p_LT": np.around(re_p_long_tail * 100, decimals=2),
        tag + "_r_LT": np.around(re_r_long_tail * 100, decimals=2),
    }
    return best_f1, output


def report(args, model, features, label_loader):
    dataloader = DataLoader(features, batch_size=args.test_batch_size,
                            shuffle=False, collate_fn=collate_fn_kd, drop_last=False)
    preds = []
    logits = []
    for batch in dataloader:
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'mention_pos': batch[5],
                  'mention_hts': batch[6],
                  'padded_mention': batch[7],
                  'padded_mention_mask': batch[8],
                  'sentid_mask': batch[9],
                  'entity_types': batch[11],
                  'segment_spans': batch[12],
                  'label_loader': label_loader,
                  }

        with torch.no_grad():
            pred, logit = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            logits.append(logit.detach().cpu())

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds, logits


def get_random_mask(train_features, drop_prob):
    new_features = []
    n_e = 42
    for old_feature in train_features:
        feature = deepcopy(old_feature)
        neg_labels = torch.tensor(feature['labels'])[:, 0]
        neg_index = torch.where(neg_labels == 1)[0]
        pos_index = torch.where(neg_labels == 0)[0]
        perm = torch.randperm(neg_index.size(0))
        sampled_negative_index = neg_index[perm[:int(
            drop_prob * len(neg_index))]]
        neg_mask = torch.ones(len(feature['labels']))
        neg_mask[sampled_negative_index] = 0
        # feature['negative_mask'] = neg_mask
        pad_neg = torch.zeros((n_e, n_e))
        num_e = int(math.sqrt(len(neg_mask)))
        pad_neg[:num_e, :num_e] = neg_mask.view(num_e, num_e)
        feature['negative_mask'] = pad_neg
        new_features.append(feature)
    return new_features


def add_logits_to_features(features, logits):
    new_features = []
    for i, old_feature in enumerate(features):
        new_feature = deepcopy(old_feature)
        new_feature['teacher_logits'] = logits[i]
        assert logits[i].shape[0] == len(new_feature['hts'])
        new_features.append(new_feature)

    return new_features


def create_negative_mask(train_features, drop_prob):
    n_e = 42
    new_features = []
    for old_feature in train_features:
        feature = deepcopy(old_feature)
        neg_labels = np.array(feature['labels'])[:, 0]
        neg_index = np.squeeze(np.argwhere(neg_labels))
        sampled_negative_index = np.random.choice(
            neg_index, int(drop_prob * len(neg_index)))
        neg_mask = np.ones(len(feature['labels']))
        neg_mask[sampled_negative_index] = 0
        neg_mask = torch.tensor(neg_mask)
        pad_neg = torch.zeros((n_e, n_e))
        num_e = int(math.sqrt(len(neg_mask)))
        pad_neg[:num_e, :num_e] = neg_mask.view(num_e, num_e)
        feature['negative_mask'] = pad_neg
        new_features.append(feature)
    return new_features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path",
                        default="bert-base-cased", type=str)

    parser.add_argument(
        "--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--save_last", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--teacher_path", default="", type=str)
    parser.add_argument("--load_pretrained", default="", type=str)
    parser.add_argument("--knowledge_distil", default="", type=str)
    parser.add_argument("--output_name", default="result.json", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--classifier_lr", default=1e-4, type=float,
                        help="The initial learning rate for Classifier.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--drop_prob", default=0.0, type=float,
                        help="Negative Sample Discard rate.")
    parser.add_argument("--gamma_pos", default=1.0, type=float,
                        help="Gamma for positive class")
    parser.add_argument("--gamma_neg", default=1.0, type=float,
                        help="Gamma for negative class")
    parser.add_argument("--drop_FP", default=0.0, type=float,
                        help="Potential FP Discard rate.")
    parser.add_argument("--drop_FN", default=0.0, type=float,
                        help="Potential FN Discard rate.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--start_steps", default=-1, type=int)
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--lambda_1", default=1.0, type=float, help="lambda_1")
    parser.add_argument("--lambda_2", default=1.0, type=float, help="lambda_2")
    parser.add_argument("--tau", default=2.0, type=float, help="tau")
    parser.add_argument("--tau_base", default=1.0, type=float, help="tau_base")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    suffix = '.{}.pt'.format(args.model_name_or_path)
    read = read_docred
    if os.path.exists(os.path.join(args.data_dir, args.train_file + suffix)):
        train_features = torch.load(os.path.join(
            args.data_dir, args.train_file + suffix))
        print('Loaded train features')
    else:
        train_file = os.path.join(args.data_dir, args.train_file)
        train_features = read(train_file, tokenizer,
                              max_seq_length=args.max_seq_length)
        torch.save(train_features, os.path.join(
            args.data_dir, args.train_file + suffix))
        print('Created and saved new train features')
    if os.path.exists(os.path.join(args.data_dir, args.dev_file + suffix)):
        dev_features = torch.load(os.path.join(
            args.data_dir, args.dev_file + suffix))
        print('Loaded dev features')
    else:
        dev_file = os.path.join(args.data_dir, args.dev_file)
        dev_features = read(dev_file, tokenizer,
                            max_seq_length=args.max_seq_length)
        torch.save(dev_features, os.path.join(
            args.data_dir, args.dev_file + suffix))
        print('Created and saved new dev features')
    if os.path.exists(os.path.join(args.data_dir, args.test_file + suffix)):
        test_features = torch.load(os.path.join(
            args.data_dir, args.test_file + suffix))
        print('Loaded test features')
    else:
        test_file = os.path.join(args.data_dir, args.test_file)
        test_features = read(test_file, tokenizer,
                             max_seq_length=args.max_seq_length)
        torch.save(test_features, os.path.join(
            args.data_dir, args.test_file + suffix))
        print('Created and saved new train features')
    n_e = 42

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    label_features = get_label_input_ids(args.data_dir, tokenizer)

    label_loader = DataLoader(label_features, batch_size=36,
                              shuffle=False, collate_fn=label_collate_fn, drop_last=False)

    set_seed(args)
    model = DocREModel_KD(args, config, model, num_labels=args.num_labels)
    model.to(0)

    if args.load_pretrained != "":  # Training from checkpoint
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_pretrained), strict=False)
        print('Loaded from checkpoint')
        dev_score, dev_output = evaluate(
            args, model, dev_features, label_loader, tag="dev")
        print(dev_output)
        train(args, model, train_features,
              dev_features, test_features, label_loader)
    elif args.knowledge_distil != "":  # KD-Pre-Training
        print('KD_pretraining')
        student_model = AutoModel.from_pretrained(args.model_name_or_path,
                                                  from_tf=bool(
                                                      ".ckpt" in args.model_name_or_path),
                                                  config=config,
                                                  )
        student_model = DocREModel_KD(
            args, config, student_model, num_labels=args.num_labels)
        student_model.to(0)
        train(args, student_model, train_features,
              dev_features, test_features, label_loader)
    elif args.teacher_path != "":  # KD inference logits
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.teacher_path), strict=False)
        dev_score, dev_output = evaluate(
            args, model, dev_features, label_loader, tag="dev")
        print(dev_output)
        pred, logits = report(args, model, train_features, label_loader)
        new_features = add_logits_to_features(train_features, logits)
        torch.save(new_features, os.path.join(
            args.data_dir, args.train_file + suffix))

        with open(args.output_name, "w") as fh:
            json.dump(pred, fh)
    elif args.load_path != "":  # Testing
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path), strict=False)
        dev_score, dev_output = evaluate(
            args, model, dev_features, label_loader, tag="dev")
        print(dev_output)
        pred, logits = report(args, model, test_features, label_loader)
        with open(args.output_name, "w") as fh:
            json.dump(pred, fh)
    else:  # Training from scratch
        train(args, model, train_features,
              dev_features, test_features, label_loader)


if __name__ == "__main__":
    main()
