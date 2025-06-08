import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from prepro import read_docred
from utils import *
from trainer import Trainer

import autoRun
autoRun.choose_gpu(n_gpus=1, retry=True, min_gpu_memory=10000, sleep_time=30)

import wandb

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path_vae", default="", type=str)
    parser.add_argument("--save_path_diff", default="", type=str)
    parser.add_argument("--load_path_vae", default="", type=str)
    parser.add_argument("--load_path_diff", default="", type=str)
    parser.add_argument("--save_last", default="", type=str)
    parser.add_argument("--load_baseline", default="", type=str)
    parser.add_argument("--output_name", default="result.json", type=str)
    parser.add_argument('--evaluation', action='store_true', default=False)

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
    parser.add_argument("--lr_vae", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--learning_rate_min_vae', type=float, default=1e-5,
                        help='min learning rate')
    parser.add_argument("--lr_diffnet", default=2e-4, type=float,
                        help="The initial learning rate for diffusion priors.")
    parser.add_argument('--learning_rate_min_diff', type=float, default=2e-5,
                        help='min learning rate of diffusion prior')
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
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Warm up epoch for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--start_steps", default=-1, type=int) 
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97, help="Number of relation types in dataset.")
    parser.add_argument("--lambda_1", default=1.0, type=float, help="lambda_1")
    parser.add_argument("--lambda_2", default=1.0, type=float, help="lambda_2")
    parser.add_argument("--tau", default=2.0, type=float, help="tau")
    parser.add_argument("--tau_base", default=1.0, type=float, help="tau_base")
    parser.add_argument("--gamma", default=1.0, type=float, help="gamma")
    parser.add_argument("--latent_size", default=32, type=int, help="The dimension of latent space")
    parser.add_argument("--kl_coeff", default=0.001, type=float, help="Coefficients of KL")

    # Diffusion process parameters
    parser.add_argument('--diffusion_steps', '-diffusion_step', type=int, default=50)
    parser.add_argument('--learn_sigma', '-learn_sigma', action='store_true', default=False)
    parser.add_argument('--sigma_small', '-sigma_small', action='store_true', default=False)
    parser.add_argument('--noise_schedule', '-noise_schedule', choices=['cosine', 'linear'], default='linear')
    parser.add_argument('--use_kl', '-use_kl', action='store_true', default=False)
    parser.add_argument('--use_ddim', '-use_ddim', action='store_true', default=False)
    parser.add_argument('--clip_denoised', '-clip_denoised', action='store_true', default=False)
    parser.add_argument('--w', '-w', type=float, default=0.1)
    parser.add_argument('--predict_xstart', '-predict_xstart', action='store_true', default=False)
    parser.add_argument('--rescale_timesteps', '-rescale_timesteps', action='store_true', default=False)
    parser.add_argument('--rescale_learned_sigmas', '-rescale_learned_sigmas', action='store_true', default=False)
    parser.add_argument('--timestep_respacing', '-timestep_respacing', type=str, default='')
    parser.add_argument('--schedule_sampler', '-schedule_sampler', choices=['uniform', 'loss-second-moment'], default='uniform')
    parser.add_argument('--latent_dir', '-latent_dir', type=str, default='./visualization/latent_samples')
    parser.add_argument('--visualize_prior', '-visualize_prior', action='store_true', default=False)
    parser.add_argument('--training', '-training', action='store_true')
    parser.add_argument('--th_prob', '-th_prob', type=float, default=0.1)
    parser.add_argument('--gen_number', '-gen_number', type=int, default=100)
    parser.add_argument('--factor', '-factor', type=int, default=10)

    # Transformers diffusion hyperparemeters
    parser.add_argument("--enc_dec_model", type=str, default="facebook/bart-base")
    parser.add_argument("--tx_dim", type=int, default=512)
    parser.add_argument("--tx_depth", type=int, default=6)
    parser.add_argument("--scale_shift", action="store_true", default=False)
    parser.add_argument("--num_dense_connections", type=int, default=0)
    parser.add_argument("--disable_dropout", action="store_true", default=False)
    parser.add_argument("--class_conditional", action="store_true", default=False)
    parser.add_argument("--class_unconditional_prob", type=float, default=.1)
    parser.add_argument("--seq2seq_unconditional_prob", type=float, default=0.1)
    parser.add_argument("--self_condition", action="store_true", default=False)
    parser.add_argument("--train_prob_self_cond", type=float, default=0.5)

    args = parser.parse_args()
    print(args)
    
    wandb.init(project="VaeDiff-DocRE")

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
        train_features = torch.load(os.path.join(args.data_dir, args.train_file + suffix))
        print('Loaded train features')
    else:
        train_file = os.path.join(args.data_dir, args.train_file)
        train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(train_features, os.path.join(args.data_dir, args.train_file + suffix))
        print('Created and saved new train features')
    if os.path.exists(os.path.join(args.data_dir, args.dev_file + suffix)):
        dev_features = torch.load(os.path.join(args.data_dir, args.dev_file + suffix))
        print('Loaded dev features')
    else:
        dev_file = os.path.join(args.data_dir, args.dev_file)
        dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(dev_features, os.path.join(args.data_dir, args.dev_file + suffix))
        print('Created and saved new dev features')
    if os.path.exists(os.path.join(args.data_dir, args.test_file + suffix)):
        test_features = torch.load(os.path.join(args.data_dir, args.test_file + suffix))
        print('Loaded test features')
    else:
        test_file = os.path.join(args.data_dir, args.test_file)
        test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(test_features, os.path.join(args.data_dir, args.test_file + suffix))
        print('Created and saved new train features')

    n_e = 42

    encoder = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    label_features = get_label_input_ids(args.data_dir, tokenizer)
    label_loader = DataLoader(label_features, batch_size=36, shuffle=False, collate_fn=label_collate_fn, drop_last=False)
    set_seed(args)

    #### TRAINING SETUP
    # Models and Diffusion Setup
    model, vae, diffusion_net, ema_diffusion = build_models(args, config, encoder)
    diffusion, schedule_sampler = build_diff_process(args)
    
    # Dataloader
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn_kd, drop_last=False)
    dev_dataloader = DataLoader(dev_features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn_kd, drop_last=False)
    test_dataloader = DataLoader(test_features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn_kd, drop_last=False)
    
    trainer = Trainer(
        args=args,
        model=model,
        vae=vae,
        diff_net=diffusion_net,
        ema_diff=ema_diffusion,
        diff_process=diffusion,
        schedule_sampler=schedule_sampler,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        test_dataloader=test_dataloader
    )
    if not args.evaluation:
        trainer.train(wandb)
    else:
        current_nll_losses = trainer.eval()
        print(f"Evaluation loss: {current_nll_losses}")


if __name__ == "__main__":
    main()