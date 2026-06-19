from diffusers import DDIMScheduler, StableDiffusionPipeline

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] loading stable diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.t_range = t_range
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings
    
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=100):
        latent_model_input = torch.cat([latents_noisy] * 2)
            
        tt = torch.cat([t] * 2)
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred


    def get_sds_loss(
        self, 
        latents,
        text_embeddings, 
        guidance_scale=100, 
        grad_scale=1,
    ):
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (latents.shape[0],),
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            noise_pred = self.get_noise_preds(
                latents_noisy, t, text_embeddings, guidance_scale
            )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = grad_scale * w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]
        return loss
    
    
    def get_pds_loss(
        self, src_latents, tgt_latents, 
        src_text_embedding, tgt_text_embedding,
        guidance_scale=7.5, 
        grad_scale=1,
    ):
        t = torch.randint(
            max(self.min_step, 1),
            self.max_step + 1,
            (1,),
            dtype=torch.long,
            device=self.device,
        )
        t_prev = t - 1

        beta_t = self.scheduler.betas[t].to(self.device)
        alpha_bar_t = self.alphas[t]
        alpha_bar_t_prev = self.alphas[t_prev]
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t).sqrt()

        noise_t = torch.randn_like(tgt_latents)
        noise_t_prev = torch.randn_like(tgt_latents)

        def stochastic_latent(latents, text_embedding):
            latents_noisy = self.scheduler.add_noise(latents, noise_t, t)
            latents_noisy_prev = self.scheduler.add_noise(latents, noise_t_prev, t_prev)

            with torch.no_grad():
                noise_pred = self.get_noise_preds(
                    latents_noisy, t, text_embedding, guidance_scale
                )

            x0_pred = (latents_noisy - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
            mu = (
                alpha_bar_t_prev.sqrt() * beta_t / (1 - alpha_bar_t)
            ) * x0_pred + (
                (1 - beta_t).sqrt() * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            ) * latents_noisy

            return (latents_noisy_prev - mu) / sigma_t

        zt_tgt = stochastic_latent(tgt_latents, tgt_text_embedding)
        zt_src = stochastic_latent(src_latents, src_text_embedding)

        grad = grad_scale * (zt_tgt - zt_src)
        grad = torch.nan_to_num(grad)

        target = (tgt_latents - grad).detach()
        loss = 0.5 * F.mse_loss(tgt_latents, target, reduction="sum") / tgt_latents.shape[0]
        return loss
    
    
    @torch.no_grad()
    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents
