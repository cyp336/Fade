import json
from collections import OrderedDict
from typing import Callable, Dict, Optional

import torch
import numpy as np
import PIL
import gc

from tqdm.auto import trange, tqdm
from diffusion_core.guiders.opt_guiders import opt_registry
from diffusion_core.diffusion_utils import latent2image, image2latent
from diffusion_core.custom_forwards.unet_sd import unet_forward
from diffusion_core.guiders.noise_rescales import noise_rescales
from diffusion_core.inversion import Inversion, NullInversion, NegativePromptInversion
from diffusion_core.utils import toggle_grad, use_grad_checkpointing
import pdb
import torch.nn.functional as F

class GuidanceEditing:
    def __init__(
            self,
            model,
            config
    ):

        self.config = config
        self.model = model
        
        #my code
        toggle_grad(self.model.transformer, False)

        if config.get('gradient_checkpointing', False):
            use_grad_checkpointing(mode=True)
        else:
            use_grad_checkpointing(mode=False)

        self.guiders = {
            g_data.name: (opt_registry[g_data.name](**g_data.get('kwargs', {})), g_data.g_scale)
            for g_data in config.guiders
        }
        #my code
        #self._setup_inversion_engine()
        self.latents_stack = []

        self.context = None

        self.noise_rescaler = noise_rescales[config.noise_rescaling_setup.type](
            config.noise_rescaling_setup.init_setup,
            **config.noise_rescaling_setup.get('kwargs', {})
        )

        for guider_name, (guider, _) in self.guiders.items():
            guider.clear_outputs()

        self.self_attn_layers_num = config.get('self_attn_layers_num', [6, 1, 9])
        if type(self.self_attn_layers_num[0]) is int:
            for i in range(len(self.self_attn_layers_num)):
                self.self_attn_layers_num[i] = (0, self.self_attn_layers_num[i])
            

    def _setup_inversion_engine(self):
        if self.config.inversion_type == 'ntinv':
            self.inversion_engine = NullInversion(
                self.model,
                self.model.scheduler.num_inference_steps,
                self.config.guiders[0]['g_scale'],
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        elif self.config.inversion_type == 'npinv':
            self.inversion_engine = NegativePromptInversion(
                self.model,
                self.model.scheduler.num_inference_steps,
                self.config.guiders[0]['g_scale'],
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        elif self.config.inversion_type == 'dummy':
            self.inversion_engine = Inversion(
                self.model,
                self.model.scheduler.num_inference_steps,
                self.config.guiders[0]['g_scale'],
                forward_guidance_scale=1,
                verbose=self.config.verbose
            )
        else:
            raise ValueError('Incorrect InversionType')

    def __call__(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
            control_image: Optional[PIL.Image.Image] = None,
            verbose: bool = False
    ):
        self.train(
            image_gt,
            inv_prompt,
            trg_prompt,
            control_image,
            verbose
        )

        return self.edit()

    def train(
            self,
            image_gt: PIL.Image.Image,
            inv_prompt: str,
            trg_prompt: str,
            control_image: Optional[PIL.Image.Image] = None,
            verbose: bool = False
    ):
        self.init_prompt(inv_prompt, trg_prompt)
        self.verbose = verbose

        image_gt = np.array(image_gt)
        if self.config.start_latent == 'inversion':
            _, self.inv_latents, self.uncond_embeddings = self.inversion_engine(
                image_gt, inv_prompt,
                verbose=self.verbose
            )
        elif self.config.start_latent == 'random':
            self.inv_latents = self.sample_noised_latents(
                image2latent(image_gt, self.model)
            )
        elif self.config.start_latent == 'latent_trajectory':
            #self.inv_latents=torch.load('/home/whl/workspace/sd3/Guide-and-Rescale/result/zebra_ddim_trajectory.pt')
            self.inv_latents=torch.load("/home/whl/workspace/sd3/Guide-and-Rescale/result/zebra_belm_trajectory.pt")
            self.uncond_embeddings=None
        else:
            raise ValueError('Incorrect start latent type')
        
        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.model_patch(self.model, self_attn_layers_num=self.self_attn_layers_num)

        self.start_latent = self.inv_latents[-1].clone()

        params = {
            'model': self.model,
            'inv_prompt': inv_prompt,
            'trg_prompt': trg_prompt
        }
        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'train'):
                guider.train(params)

        for guider_name, (guider, _) in self.guiders.items():
            guider.clear_outputs()

    def _construct_data_dict(
            self, latents,
            diffusion_iter,
            timestep
    ):
        uncond_emb, inv_prompt_emb, trg_prompt_emb = self.context.chunk(3)

        if self.uncond_embeddings is not None:
            uncond_emb = self.uncond_embeddings[diffusion_iter]

        data_dict = {
            'latent': latents,
            'inv_latent': self.inv_latents[-diffusion_iter - 1],
            'timestep': timestep,
            'model': self.model,
            'uncond_emb': uncond_emb,
            'trg_emb': trg_prompt_emb,
            'inv_emb': inv_prompt_emb,
            'diff_iter': diffusion_iter
        }

        with torch.no_grad():
            uncond_unet = unet_forward(
                self.model,
                data_dict['latent'],
                data_dict['timestep'],
                data_dict['uncond_emb'],
                None
            )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.clear_outputs()

        with torch.no_grad():
            inv_prompt_unet = unet_forward(
                self.model,
                data_dict['inv_latent'],
                data_dict['timestep'],
                data_dict['inv_emb'],
                None
            )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'inv_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_inv_inv": guider.output})
                guider.clear_outputs()

        data_dict['latent'].requires_grad = True

        src_prompt_unet = unet_forward(
            self.model,
            data_dict['latent'],
            data_dict['timestep'],
            data_dict['inv_emb'],
            None
        )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_inv": guider.output})
                guider.clear_outputs()

        trg_prompt_unet = unet_forward(
            self.model,
            data_dict['latent'],
            data_dict['timestep'],
            data_dict['trg_emb'],
            None
        )

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_trg' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_trg": guider.output})
                guider.clear_outputs()

        data_dict.update({
            'uncond_unet': uncond_unet,
            'trg_prompt_unet': trg_prompt_unet,
        })

        return data_dict
    
    #my code
    def _construct_data_dict_sd3(
            self, latents,
            diffusion_iter,
            timestep
    ):
        uncond_emb, inv_prompt_emb, trg_prompt_emb = self.uncond_embed, self.inv_prompt_embed, self.trg_prompt_embed

        if self.uncond_embeddings is not None:
            uncond_emb = self.uncond_embeddings[diffusion_iter]
        
        timestep = timestep.expand(latents.shape[0])
        data_dict = {
            'latent': latents,
            'inv_latent': self.inv_latents[-diffusion_iter - 1],
            'timestep': timestep,
            'model': self.model,
            'uncond_emb': uncond_emb,
            'trg_emb': trg_prompt_emb,
            'inv_emb': inv_prompt_emb,
            'diff_iter': diffusion_iter
        }
 
        with torch.no_grad():
            uncond_transformer = self.model.transformer(
                    hidden_states=data_dict['latent'],
                    timestep=data_dict['timestep'],
                    encoder_hidden_states=data_dict['uncond_emb'],
                    pooled_projections=self.uncond_embed_pooled,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.clear_outputs()

        with torch.no_grad():
            inv_prompt_transformer = self.model.transformer(
                    hidden_states=data_dict['inv_latent'],
                    timestep=data_dict['timestep'],
                    encoder_hidden_states=data_dict['inv_emb'],
                    pooled_projections=self.inv_prompt_embed_pooled,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'inv_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_inv_inv": guider.output})
                guider.clear_outputs()

        data_dict['latent'].requires_grad = True
        
        src_prompt_transformer = self.model.transformer(
                hidden_states=data_dict['latent'],
                timestep=data_dict['timestep'],
                encoder_hidden_states=data_dict['inv_emb'],
                pooled_projections=self.inv_prompt_embed_pooled,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
        

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_inv": guider.output})
                guider.clear_outputs()
        
        trg_prompt_transformer = self.model.transformer(
                hidden_states=data_dict['latent'],
                timestep=data_dict['timestep'],
                encoder_hidden_states=data_dict['trg_emb'],
                pooled_projections=self.trg_prompt_embed_pooled,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_trg' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_trg": guider.output})
                guider.clear_outputs()

        data_dict.update({
            'uncond_unet': uncond_transformer,
            'trg_prompt_unet': trg_prompt_transformer,
        })

        return data_dict
    
    def _get_noise(self, data_dict, diffusion_iter):
        backward_guiders_sum = 0.
        noises = {
            'uncond': data_dict['uncond_unet'],
        }
        index = torch.where(self.model.scheduler.timesteps == data_dict['timestep'])[0].item()

        # self.noise_rescaler
        for name, (guider, g_scale) in self.guiders.items():
            if guider.grad_guider:
                cur_noise_pred = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                noises[name] = cur_noise_pred
            else:
                energy = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                if not torch.allclose(energy, torch.tensor(0.)):
                    backward_guiders_sum += energy

        if hasattr(backward_guiders_sum, 'backward'):
            backward_guiders_sum.backward()
            noises['other'] = data_dict['latent'].grad

        scales = self.noise_rescaler(noises, index)
        noise_pred = sum(scales[k] * noises[k] for k in noises)

        for g_name, (guider, _) in self.guiders.items():
            if not guider.grad_guider:
                guider.clear_outputs()
            gc.collect()
            torch.cuda.empty_cache()

        return noise_pred

    def _get_noise_sd3(self, data_dict, diffusion_iter):
        backward_guiders_sum = 0.
        noises = {
            'uncond': data_dict['uncond_unet'],
        }
        index = torch.where(self.model.scheduler.timesteps == data_dict['timestep'])[0].item()

        # self.noise_rescaler
        for name, (guider, g_scale) in self.guiders.items():
            if guider.grad_guider:
                cur_noise_pred = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                noises[name] = cur_noise_pred
            else:
                energy = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                if not torch.allclose(energy, torch.tensor(0.)):
                    backward_guiders_sum += energy

        if hasattr(backward_guiders_sum, 'backward'):
            backward_guiders_sum.backward()
            noises['other'] = data_dict['latent'].grad

        scales = self.noise_rescaler(noises, index)
        noise_pred = sum(scales[k] * noises[k] for k in noises)

        for g_name, (guider, _) in self.guiders.items():
            if not guider.grad_guider:
                guider.clear_outputs()
            gc.collect()
            torch.cuda.empty_cache()

        return noise_pred

    @staticmethod
    def _get_scale(g_scale, diffusion_iter):
        if type(g_scale) is float:
            return g_scale
        else:
            return g_scale[diffusion_iter]

    @torch.no_grad()
    def _step(self, noise_pred, t, latents):
        latents = self.model.scheduler.step_backward(noise_pred, t, latents).prev_sample
        self.latents_stack.append(latents.detach())
        return latents
    
    @torch.no_grad()
    def _step_sd3(self, noise_pred, t, latents):
        latents = self.model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        self.latents_stack.append(latents.detach())
        return latents

    @torch.no_grad()
    def _step_sd3_belm(self, noise_pred, t, latents):
        latents = self.model.scheduler.step_belm_2(noise_pred, t, latents, return_dict=False)[0]
        self.latents_stack.append(latents.detach())
        return latents

    def edit(self):
        #my code
        #self.model.scheduler.set_timesteps(self.model.scheduler.num_inference_steps)
        
        latents = self.start_latent
        self.latents_stack = []

        for i, timestep in tqdm(
                enumerate(self.model.scheduler.timesteps),
                
                #my code
                #total=self.model.scheduler.num_inference_steps,
                total=len(self.model.scheduler.timesteps),  
                          
                desc='Editing',
                disable=not self.verbose
        ):
            # 1. Construct dict            
            #data_dict = self._construct_data_dict(latents, i, timestep)
            data_dict = self._construct_data_dict_sd3(latents, i, timestep)

            # 2. Calculate guidance
            #noise_pred = self._get_noise(data_dict, i)
            noise_pred = self._get_noise_sd3(data_dict, i)

            # 3. Scheduler step
            #important whether use the noise_pred_target or not 
            #latents = self._step(noise_pred, timestep, latents)
            #latents = self._step_sd3(noise_pred, timestep, latents)
            
            latents = self._step_sd3_belm(noise_pred, timestep, latents)

        self._model_unpatch(self.model)
        return latent2image(latents, self.model)[0]

    @torch.no_grad()
    def init_prompt(self, inv_prompt: str, trg_prompt: str):
        
        #my code
        # trg_prompt_embed = self.get_prompt_embed(trg_prompt)
        # inv_prompt_embed = self.get_prompt_embed(inv_prompt)
        # uncond_embed = self.get_prompt_embed("")
        # self.context = torch.cat([uncond_embed, inv_prompt_embed, trg_prompt_embed])
        
        self.uncond_embed,self.trg_prompt_embed,self.uncond_embed_pooled,self.trg_prompt_embed_pooled = self.get_prompt_embed(trg_prompt)
        _,self.inv_prompt_embed,_,self.inv_prompt_embed_pooled = self.get_prompt_embed(inv_prompt)
        

    # def get_prompt_embed(self, prompt: str):
    #     text_input = self.model.tokenizer(
    #         [prompt],
    #         padding="max_length",
    #         max_length=self.model.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     text_embeddings = self.model.text_encoder(
    #         text_input.input_ids.to(self.model.device)
    #     )[0]

    #     return text_embeddings
    
    #my code 
    def get_prompt_embed(self, prompt: str):
        #important : device
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds=self.model.encode_prompt(prompt,prompt_2=None,prompt_3=None,
                                                                                                     device="cuda")
        return negative_prompt_embeds,prompt_embeds,negative_pooled_prompt_embeds,pooled_prompt_embeds


    def sample_noised_latents(self, latent):
        all_latent = [latent.clone().detach()]
        latent = latent.clone().detach()
        for i in trange(self.model.scheduler.num_inference_steps, desc='Latent Sampling'):
            timestep = self.model.scheduler.timesteps[-i - 1]
            if i + 1 < len(self.model.scheduler.timesteps):
                next_timestep = self.model.scheduler.timesteps[- i - 2]
            else:
                next_timestep = 999

            alpha_prod_t = self.model.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_timestep]

            alpha_slice = alpha_prod_t_next / alpha_prod_t

            latent = torch.sqrt(alpha_slice) * latent + torch.sqrt(1 - alpha_slice) * torch.randn_like(latent)
            all_latent.append(latent)
        return all_latent

    def _model_unpatch(self, model):
        def new_forward_info(self):
            def patched_forward(
                    hidden_states,              #[1,4096,1536]         
                    encoder_hidden_states=None, #[1,154,1536]
                    attention_mask=None,
            ):
                residual = hidden_states

                input_ndim = hidden_states.ndim
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                context_input_ndim = encoder_hidden_states.ndim
                if context_input_ndim == 4:
                    batch_size, channel, height, width = encoder_hidden_states.shape
                    encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size = encoder_hidden_states.shape[0]

                # `sample` projections.
                query = self.to_q(hidden_states) #[1,4096,1536]
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)

                # `context` projections.
                encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                # attention
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1) #[1,4250,1536]
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // self.heads
                query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) #[1,24,4250,64]
                key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False) #[1,24,4250,64]
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype) #[1,4250,1536]

                # Split the attention outputs.
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : residual.shape[1]],
                    hidden_states[:, residual.shape[1] :],
                ) #[1,4096,1536] [1,154,1536] 
                  
                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)
                if not self.context_pre_only:
                    encoder_hidden_states = self.to_add_out(encoder_hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
                if context_input_ndim == 4:
                    encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                return hidden_states, encoder_hidden_states
            return patched_forward
    
        def register_attn(module):
            if 'Attention' in module.__class__.__name__:
                module.forward = new_forward_info(module)
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    register_attn(module_)

        def remove_hooks(module):
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks: Dict[int, Callable] = OrderedDict()
            if hasattr(module, 'children'):
                for module_ in module.children():
                    remove_hooks(module_)
        #my code
        register_attn(model.transformer)
        remove_hooks(model.transformer)
