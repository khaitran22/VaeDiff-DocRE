import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
from functools import partial
from utils import *

class Trainer:
    def __init__(self, args, model, vae, diff_net, ema_diff, diff_process, schedule_sampler,
                 train_dataloader, dev_dataloader, test_dataloader):
        self.args = args
        self.model = model
        self.vae = vae
        self.diff_net = diff_net
        self.ema_diff = ema_diff
        self.diff_process = diff_process
        self.schedule_sampler = schedule_sampler
        
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader

        self._training_setup()
    
    def _training_setup(self):
        self.optimizer_vae = AdamW(self.vae.parameters(), lr=self.args.lr_vae, eps=self.args.adam_epsilon)
        self.vae_scheduler = CosineAnnealingLR(self.optimizer_vae, float(self.args.num_train_epochs - self.args.warmup_epochs - 1), eta_min=self.args.learning_rate_min_vae)
        self.optimizer_diff = AdamW(self.diff_net.parameters(), lr=self.args.lr_diffnet, eps=self.args.adam_epsilon)
    
    def _pair_encode(self, model, input_ids, labels, attention_mask, entity_pos, hts):
        sequence_output, attention = model.encode(input_ids, attention_mask)
        bs = sequence_output.size(0)
        ne = 42
        nes = [len(x) for x in entity_pos]
        hs_e, rs_e, ts_e = model.get_hrt(sequence_output, attention, entity_pos, hts)

        hs_e_1 = torch.tanh(model.head_extractor(torch.cat([hs_e, rs_e], dim=3)))
        ts_e_1 = torch.tanh(model.tail_extractor(torch.cat([ts_e, rs_e], dim=3))) 
        
        b1_e = hs_e_1.view(bs, ne, ne, model.emb_size // model.block_size, model.block_size)
        b2_e = ts_e_1.view(bs, ne, ne, model.emb_size // model.block_size, model.block_size)
        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(bs, ne, ne, model.emb_size * model.block_size)

        feature = model.projection(bl_e)
        feature = model.axial_transformer(feature) + feature
        feature = torch.cat([feature.clone()[x, :nes[x], :nes[x] , :].reshape(-1, model.config.hidden_size) for x in range(len(nes))])

        labels = [torch.tensor(label) for label in labels]
        labels = torch.cat(labels, dim=0).to(rs_e.device)

        return feature, labels
    
    def _embed_pair(self, model, input_ids, labels, attention_mask, entity_pos, hts):
        model.eval()
        with torch.no_grad():
            feature, target_labels = self._pair_encode(model, input_ids, labels, attention_mask, entity_pos, hts)
        return feature, target_labels
    
    def train(self, wandb):
        train_iterator = range(int(self.args.num_train_epochs))
        print("Total epochs: {}".format(self.args.num_train_epochs))
        print("Warmup epochs: {}".format(self.args.warmup_epochs))
        num_steps = 0
        best_nll_score = 1e+20
        
        for epoch in train_iterator:
            with tqdm(self.train_dataloader) as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                self.vae.train(); self.diff_net.train()

                for step, batch in enumerate(tepoch):

                    ##############################################
                    ###### Retrieve positive entity pairs ########
                    ##############################################
                    inputs = {
                        'model':self.model,
                        'input_ids': batch[0].to(self.args.device),
                        'attention_mask': batch[1].to(self.args.device),
                        'labels': batch[2], 
                        'entity_pos': batch[3],
                        'hts': batch[4],
                    }
                    device = inputs['input_ids'].device
                    feature, labels = self._embed_pair(**inputs)
                    mask = labels[:, 0] != 1
                    feature = feature[mask, :]
                    labels = labels[mask, :]
                    
                    ##############################################
                    ###### Update the VAE encoder/decoder ########
                    ##############################################
                    bs, _ = feature.size()
                    logits, all_p_latent, all_log_p = self.vae(feature)
                    output = self.vae.decoder_output(logits)
                    vae_recon_loss = self.vae.reconstruction_loss(output, feature)
                    
                    model_kwargs = {
                        'class_id': labels
                    }
                    t, weights = self.schedule_sampler.sample(bs, device)
                    compute_losses = partial(
                        self.diff_process.training_losses,
                        self.diff_net,
                        all_p_latent,
                        t,
                        True,
                        model_kwargs=model_kwargs,
                    )
                    losses = compute_losses()
                    diff_loss = losses['loss']
                    
                    cross_entropy_per_var = weights.unsqueeze(-1) * diff_loss.unsqueeze(-1)
                    all_neg_log_p = cross_entropy_per_var + self.diff_process.cross_entropy_const(torch.zeros_like(t, dtype=t.dtype, device=t.device), all_p_latent)
                    kl_all_list, kl_vals_per_group, kl_diag_list = kl_per_group_vada(all_log_p, all_neg_log_p)
                    kl_coeff = self.args.kl_coeff if epoch >= self.args.warmup_epochs else 1e-6
                    balanced_kl, kl_coeffs, kl_vals = kl_balancer(kl_all_list, kl_coeff, kl_balance=False)
                    
                    loss_vae = (vae_recon_loss + balanced_kl) / bs
                    
                    if self.args.gradient_accumulation_steps > 1:
                        loss_vae = loss_vae / self.args.gradient_accumulation_steps
                    loss_vae.backward()
                    if step % self.args.gradient_accumulation_steps == 0:
                        if self.args.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.args.max_grad_norm)
                    self.optimizer_vae.step()
                    
                    if epoch >= self.args.warmup_epochs:
                        self.vae_scheduler.step()
                        ####################################
                        ######  Update the SGM prior #######
                        ####################################
                        self.optimizer_diff.zero_grad()
                        all_p_latent = all_p_latent.detach()
                        model_kwargs = {
                            'class_id': labels,
                            'training': True,
                            'p': self.args.th_prob
                        }
                        t, weights = self.schedule_sampler.sample(bs, device)
                        compute_losses = partial(
                            self.diff_process.training_losses,
                            self.diff_net,
                            all_p_latent,
                            t,
                            True,
                            model_kwargs=model_kwargs,
                        )
                        losses = compute_losses()
                        diff_loss = losses['loss'].mean()
                        if self.args.gradient_accumulation_steps > 1:
                            diff_loss = diff_loss / self.args.gradient_accumulation_steps
                        diff_loss.backward()
                        
                        if step % self.args.gradient_accumulation_steps == 0:
                            if self.args.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.diff_net.parameters(), self.args.max_grad_norm)
                        self.optimizer_diff.step()
                        
                        # update ema model
                        self.ema_diff.update()
                    
                    ###### log reconstr_loss and kl_pst_z_prior_z to see ######
                    loss_vae_val = loss_vae.detach().item()
                    diff_loss_val = diff_loss.detach().item() if epoch >= self.args.warmup_epochs else 10
                    wandb.log(
                        {
                            "Loss VAE": loss_vae_val,
                            "Diffusion Prior Loss": diff_loss_val,
                        }, step=num_steps)
                    num_steps += 1

                    ###### validation ######
                    if step == len(self.train_dataloader) - 1:
                        best_nll_score = self._epoch_evaluation(epoch, wandb, num_steps, best_nll_score)
                        
                        
    def _epoch_evaluation(self, epoch, wandb, num_steps, best_nll_score):
        self.vae.eval(); self.ema_diff.ema_model.eval(); self.diff_net.eval()
        nll_losses = []
        batch_sizes = []
        
        for step, batch in enumerate(tqdm(self.dev_dataloader, desc='Testing')):
            with torch.no_grad():
                inputs = {
                    'model': self.model,
                    'input_ids': batch[0].to(self.args.device),
                    'attention_mask': batch[1].to(self.args.device),
                    'labels': batch[2], 
                    'entity_pos': batch[3],
                    'hts': batch[4],
                }
                device = inputs['input_ids'].device
                
                feature, labels = self._embed_pair(**inputs)
                mask = labels[:, 0] != 1
                feature = feature[mask, :]
                labels = labels[mask, :]
                
                bs, _ = feature.size()
                logits, all_p_latent, all_log_p = self.vae(feature)
                output = self.vae.decoder_output(logits)
                vae_recon_loss = self.vae.reconstruction_loss(output, feature)

                model_kwargs = {
                    'class_id': labels
                }
                t, weights = self.schedule_sampler.sample(bs, device)
                compute_losses = partial(
                    self.diff_process.training_losses,
                    self.ema_diff.ema_model,
                    all_p_latent,
                    t,
                    True,
                    model_kwargs=model_kwargs,
                )
                losses = compute_losses()
                diff_loss = losses['loss'].mean()
    
                cross_entropy_per_var = weights.unsqueeze(-1) * diff_loss.unsqueeze(-1)
                all_neg_log_p = cross_entropy_per_var + self.diff_process.cross_entropy_const(torch.zeros_like(t, dtype=t.dtype, device=t.device), all_p_latent)
                kl_all_list, kl_vals_per_group, kl_diag_list = kl_per_group_vada(all_log_p, all_neg_log_p)
                kl_coeff = self.args.kl_coeff if epoch >= self.args.warmup_epochs else 1e-6
                balanced_kl, kl_coeffs, kl_vals = kl_balancer(kl_all_list, kl_coeff, kl_balance=False)
                
                loss_vae = (vae_recon_loss + balanced_kl) / bs
                nll_losses.append(loss_vae.item())
                batch_sizes.append(bs)
        
        dev_nll_losses = np.array(nll_losses).sum()
        dev_batch_size = np.array(batch_sizes).sum()
        current_nll_losses = dev_nll_losses / dev_batch_size
        wandb.log(
            {
                "Dev Loss NLL": current_nll_losses,
            }, 
            step=num_steps
        )
        if current_nll_losses < best_nll_score:
            best_nll_score = current_nll_losses
            print(f"Epoch {epoch+1}: {current_nll_losses} - Step: {num_steps} - Best loss!")
        else:
            print(f"Epoch {epoch+1}: {current_nll_losses} - Step: {num_steps}")
        torch.save(self.vae.state_dict(), self.args.save_path_vae)
        torch.save(self.ema_diff.state_dict(), self.args.save_path_diff)
        print("-------------------")
        return best_nll_score