import json
from collections import OrderedDict
from typing import Callable, Dict, Optional

import torch
import numpy as np
import PIL
import gc

from tqdm.auto import trange, tqdm
from diffusion_core.guiders.opt_guiders_cogvideo import opt_registry
from diffusion_core.diffusion_utils import latent2image, image2latent
from diffusion_core.custom_forwards.unet_sd import unet_forward
from diffusion_core.guiders.noise_rescales import noise_rescales
from diffusion_core.inversion import Inversion, NullInversion, NegativePromptInversion
from diffusion_core.utils import use_deterministic
from diffusion_core.utils import toggle_grad, use_grad_checkpointing
import pdb
import torch.nn.functional as F
import imageio
import cv2
import matplotlib.pyplot as plt

def latent2video(pipe,latent,output_path,fps=4,target_size = (640,640)):
    
    latent = latent.permute(0,2,1,3,4)
    latent = latent / pipe.vae.config.scaling_factor
    with torch.no_grad():
        decoded_frames = pipe.vae.decode(latent).sample
        
    frames = decoded_frames[0].squeeze(0).permute(1, 2, 3, 0).to(torch.float32).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)
    
    # resized_frames = []
    # for frame in frames:
    #     resized_frame = cv2.resize(frame, target_size,interpolation=cv2.INTER_LANCZOS4)  # Resize to (width, height)
    #     resized_frames.append(resized_frame)
    # resized_frames = np.array(resized_frames)    
    resized_frames = frames
    
    writer = imageio.get_writer(output_path, fps=fps)

    for frame in resized_frames:
        writer.append_data(frame)
    writer.close()

    return decoded_frames
    
class GuidanceEditing:
    def __init__(
            self,
            model,
            config,
            lb
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
                
        #my code
        #important
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.local_blend = lb      

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
        #my code
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
        
        #my code
        # image_gt = np.array(image_gt)
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
            #self.inv_latents=torch.load("/home/whl/workspace/result/rabbit_large_ddim_previous_text_nocfg_10m19d_float32_all_latent.pt")
            #self.inv_latents=torch.load("/home/whl/workspace/result/rabbit_large_ddim_previous_text_nocfg_10m19d_float16_all_latent.pt")
            #self.inv_latents = torch.load("/home/whl/workspace/result/bird_forest_ddim_previous_text_nocfg_10m19d_float16_all_latent_no_resize.pt")
            #self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/rabbit_resize.pt")
            # self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/rabbit.pt")
            # self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/bird_forest_resize.pt")
            #self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/cat_flower.pt")
            #self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/cat_flower_longtext_no_motion.pt")
            #self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/cat_flower_longtext_51.pt")
            
            
            #self.inv_latents = torch.load("/home/whl/workspace/fatezero/output/all_latent_ddim/swan_swarov.pt")
            #self.inv_latents = torch.load("/home/whl/workspace/fatezero/output/all_latent_ddim/swan_swarov_5b.pt")
            #self.inv_latents = torch.load("/home/whl/workspace/fatezero/output/all_latent_ddim/bear_tiger_lion_leopard_5b.pt",map_location='cuda:0')
            self.inv_latents = torch.load("/home/cby/cogvideo_code/inversion_output/all_latent_ddim/gray_dog.pt",map_location='cuda:0')
            
            # self.mask = torch.load("/home/whl/workspace/fatezero/output/mask/swan_swarov_lastblock.pt")
            # mask_f,mask_h,mask_w=self.mask.shape
            # self.mask = self.mask.gt(0.5)
            # plt.imshow(self.mask[2])
            # plt.savefig("/home/whl/workspace/cogvideo_edit/Guide-and-Rescale/mask_show_2.jpg")
            # self.mask = self.mask.to(self.device)
            # self.mask = self.mask.reshape(1,mask_f,1,mask_h,mask_w)
            
            #all_mask = torch.load("/home/whl/workspace/cogvideo_edit/metrics_mask/child_new.pt") #[9,480,720]
            all_mask = torch.load("/home/cby/cogvideo_code/mask/gray_dog.pt") #[9,480,720]
            mask = F.max_pool2d(all_mask.float(), kernel_size=(8,8), stride=(8,8))
            #mask = F.interpolate(all_mask.unsqueeze(0).float(), size=(60, 90), mode='bilinear', align_corners=False)
            mask= mask.reshape(3, 3, 60, 90)
            
            mask = mask.max(dim=1)[0]
            #mask = mask[:,1]
            
            kernel = np.ones((3, 3), np.uint8) 
            for i in range(mask.shape[0]):
                item = mask[i].detach().cpu().numpy()*255
                item = item.astype(np.uint8)
                item = cv2.dilate(item, kernel, iterations=1)
                #item = cv2.erode(item, kernel, iterations=1)
                item = torch.tensor(item/255)
                mask[i] = item
            
            mask = mask.bool()
            #mask = mask.float()
            
            for i in range(mask.shape[0]):
                plt.imshow(mask[i], cmap = "hot")
                plt.savefig(f"/home/cby/cogvideo_code/mask_visualize/gray_dog_{i}.png")
            
            self.mask = mask.reshape([1,mask.shape[0],1,mask.shape[1],mask.shape[2]])
            self.mask = self.mask.to(self.device)             
            
            
            #self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/cat_flower_longtext_no_motion.pt")
            #self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/tiger_longtext.pt")
            #self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/child_bike_longtext.pt")
            # self.inv_latents = torch.load("/home/whl/workspace/output_edit_pre/all_latent_ddim/cat_flower_longtext_new.pt")
            self.inv_latents=[item.to(self.dtype) for item in self.inv_latents]
            #self.inv_latents=torch.load("/home/whl/workspace/sd3/Guide-and-Rescale/result/zebra_belm_trajectory.pt")
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
    
    def _construct_data_dict_cogvideo(
            self, latents,
            diffusion_iter,
            timestep
    ):
        uncond_emb, inv_prompt_emb, trg_prompt_emb = self.context.chunk(3)

        if self.uncond_embeddings is not None:
            uncond_emb = self.uncond_embeddings[diffusion_iter]
        
        timestep = timestep.expand(latents.shape[0])
        #be careful
        # data_dict = {
        #     'latent': latents,
        #     'inv_latent': self.inv_latents[-diffusion_iter - 1],
        #     'timestep': timestep,
        #     'model': self.model,
        #     'uncond_emb': uncond_emb,
        #     'trg_emb': trg_prompt_emb,
        #     'inv_emb': inv_prompt_emb,
        #     'diff_iter': diffusion_iter,
        # }
        data_dict = {
            "latent": latents,
            "inv_latent": self.inv_latents[-diffusion_iter - 1],
            "timestep": timestep,
            "model": self.model,
            "uncond_emb": uncond_emb,
            "trg_emb": trg_prompt_emb,
            "inv_emb": inv_prompt_emb,
            "diff_iter": diffusion_iter,
            "inv_latent_prev": self.inv_latents[-diffusion_iter - 2]
        }
        

        image_rotary_emb = (
            self.model._prepare_rotary_positional_embeddings(480, 720, 3, self.device)
            if self.model.transformer.config.use_rotary_positional_embeddings
            else None
        )
                  
        with torch.no_grad():
            uncond_transformer = self.model.transformer(
                    hidden_states=data_dict['latent'],
                    timestep=data_dict['timestep'],
                    encoder_hidden_states=data_dict['uncond_emb'],
                    #image_rotary_emb=None,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.clear_outputs()
        
        #print(f"now {torch.cuda.memory_allocated()/1024**3} max {torch.cuda.max_memory_allocated()/1024**3}")

        with torch.no_grad():
            inv_prompt_transformer = self.model.transformer(
                    hidden_states=data_dict['inv_latent'],
                    timestep=data_dict['timestep'],
                    encoder_hidden_states=data_dict['inv_emb'],
                    #image_rotary_emb=None,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'inv_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_inv_inv": guider.output})
                #my code
                else:
                    del guider.output
                    gc.collect()
                    torch.cuda.empty_cache()     
                guider.clear_outputs()
        #print(f"now {torch.cuda.memory_allocated()/1024**3} max {torch.cuda.max_memory_allocated()/1024**3}")
        #my code
        data_dict['latent'] = data_dict['latent'].detach()
        data_dict['latent'].requires_grad = True
        
        
        src_prompt_transformer = self.model.transformer(
                hidden_states=data_dict['latent'],
                timestep=data_dict['timestep'],
                encoder_hidden_states=data_dict['inv_emb'],
                #image_rotary_emb=None,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
        

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_inv": guider.output})
                #my code
                else:
                    del guider.output
                    gc.collect()
                    torch.cuda.empty_cache()   
                guider.clear_outputs()
        #print(f"now {torch.cuda.memory_allocated()/1024**3} max {torch.cuda.max_memory_allocated()/1024**3}")
        #my code
        with torch.no_grad():      
            trg_prompt_transformer = self.model.transformer(
                    hidden_states=data_dict['latent'],
                    timestep=data_dict['timestep'],
                    encoder_hidden_states=data_dict['trg_emb'],
                    #image_rotary_emb=None,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_trg' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_trg": guider.output})
                else:
                    del guider.output
                    gc.collect()
                    torch.cuda.empty_cache() 
                guider.clear_outputs()
        #print(f"now {torch.cuda.memory_allocated()/1024**3} max {torch.cuda.max_memory_allocated()/1024**3}")
        
        data_dict.update({
            'uncond_unet': uncond_transformer,
            'trg_prompt_unet': trg_prompt_transformer,
        })
        
        return data_dict
    
    def _construct_data_dict_cogvideo_negative(
            self, latents,
            diffusion_iter,
            timestep
    ):
        uncond_emb, inv_prompt_emb, trg_prompt_emb = self.context.chunk(3)

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
                    image_rotary_emb=None,
                    return_dict=False,
                )[0]

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                guider.clear_outputs()
        
        #print(f"now {torch.cuda.memory_allocated()/1024**3} max {torch.cuda.max_memory_allocated()/1024**3}")

        with torch.no_grad():
            inv_prompt_transformer = self.model.transformer(
                    hidden_states=data_dict['inv_latent'],
                    timestep=data_dict['timestep'],
                    encoder_hidden_states=data_dict['inv_emb'],
                    image_rotary_emb=None,
                    return_dict=False,
                )[0]

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'inv_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_inv_inv": guider.output})
                #my code
                else:
                    del guider.output
                    gc.collect()
                    torch.cuda.empty_cache()     
                guider.clear_outputs()
        #print(f"now {torch.cuda.memory_allocated()/1024**3} max {torch.cuda.max_memory_allocated()/1024**3}")
        data_dict['latent'].requires_grad = True
        
        
        src_prompt_transformer = self.model.transformer(
                hidden_states=data_dict['latent'],
                timestep=data_dict['timestep'],
                encoder_hidden_states=data_dict['inv_emb'],
                image_rotary_emb=None,
                return_dict=False,
            )[0]
        

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_inv' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_inv": guider.output})
                #my code
                else:
                    del guider.output
                    gc.collect()
                    torch.cuda.empty_cache()   
                guider.clear_outputs()
        #my code
        with torch.no_grad():      
            trg_prompt_transformer = self.model.transformer(
                    hidden_states=data_dict['latent'],
                    timestep=data_dict['timestep'],
                    encoder_hidden_states=data_dict['trg_emb'],
                    image_rotary_emb=None,
                    return_dict=False,
                )[0]

        for g_name, (guider, _) in self.guiders.items():
            if hasattr(guider, 'model_patch'):
                if 'cur_trg' in guider.forward_hooks:
                    data_dict.update({f"{g_name}_cur_trg": guider.output})
                else:
                    del guider.output
                    gc.collect()
                    torch.cuda.empty_cache() 
                guider.clear_outputs()
        #print(f"now {torch.cuda.memory_allocated()/1024**3} max {torch.cuda.max_memory_allocated()/1024**3}")
        
        data_dict.update({
            "uncond_unet": uncond_transformer,
            "src_prompt_unet": src_prompt_transformer,
            "trg_prompt_unet": trg_prompt_transformer,
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
    
    def _get_noise_cogvideo_new(self, data_dict, diffusion_iter):
        noises = {
            'uncond': data_dict['uncond_unet'],
        }
        index = torch.where(self.model.scheduler.timesteps == data_dict['timestep'])[0].item()

        # self.noise_rescaler
        #note only two guider cfg+attn_map_appearance
        for name, (guider, g_scale) in self.guiders.items():
            if guider.grad_guider:
                cur_noise_pred = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                noises[name] = cur_noise_pred
            else:
                #energy_attn, energy_app = self._get_scale(g_scale, diffusion_iter) * guider(data_dict)
                energy_attn, energy_app = guider(data_dict)
                energy =energy_attn+energy_app
                if not torch.allclose(energy.to(torch.float32), torch.tensor(0.)):
                    
                    energy_attn.backward(retain_graph=True)
                    energy_attn_noise = data_dict["latent"].grad.clone()
                    data_dict["latent"].grad.zero_()
                    energy_app.backward()
                    energy_app_noise = data_dict["latent"].grad.clone()
                    
                    if diffusion_iter ==1:
                        cfg_noise_norm = torch.norm(noises['cfg']).item()
                        app_noise_norm = torch.norm(energy_app_noise).item()
                        energy_app_noise = 0.5 * cfg_noise_norm/app_noise_norm * energy_app_noise
                        attn_noise_norm = torch.norm(energy_attn_noise).item()
                        app_noise_norm = torch.norm(energy_app_noise).item()
                        ratio = app_noise_norm/attn_noise_norm
                        print(f"app_noise_norm/attn_noise_norm {ratio}")
                        energy_attn_noise = ratio * energy_attn_noise
                    
                    noises["other"] = energy_attn_noise + energy_app_noise
                    
                    # if hasattr(backward_guiders_sum, "backward"):
                    #     backward_guiders_sum.backward()
                    #     noises["other"] = data_dict["latent"].grad
                    #     #print(f"noises[other] {data_dict['latent'].grad.flatten()[:30]}")

        scales = self.noise_rescaler(noises, index)
        noise_pred = sum(scales[k] * noises[k] for k in noises)
        for g_name, (guider, _) in self.guiders.items():
            if not guider.grad_guider:
                guider.clear_outputs()
                gc.collect()
                torch.cuda.empty_cache()

        #my code
        del noises
        gc.collect()
        torch.cuda.empty_cache()            
        
        return noise_pred
    
    def _get_noise_cogvideo(self, data_dict, diffusion_iter):
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
                if not torch.allclose(energy.to(torch.float32), torch.tensor(0.)):
                    backward_guiders_sum += energy

        if hasattr(backward_guiders_sum, "backward"):
            backward_guiders_sum.backward()
            noises["other"] = data_dict["latent"].grad
            #print(f"noises[other] {data_dict['latent'].grad.flatten()[:30]}")
        
        scales = self.noise_rescaler(noises, index)
        noise_pred = sum(scales[k] * noises[k] for k in noises)
        for g_name, (guider, _) in self.guiders.items():
            if not guider.grad_guider:
                guider.clear_outputs()
                gc.collect()
                torch.cuda.empty_cache()

        #my code
        del noises
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
    
    #important
    @torch.no_grad()
    def _step_cogvideo(self, noise_pred, t, latents):
        latents = self.model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        self.latents_stack.append(latents.detach())
        return latents
    
    def _step_cogvideo_5b(self,noise_pred,t,latents,old_pred_original_sample,timestep_back):
        latents, old_pred_original_sample = self.model.scheduler.step(
            noise_pred,
            old_pred_original_sample,
            t,
            timestep_back,
            latents,
            #**extra_step_kwargs,
            return_dict=False,
        )
        return latents, old_pred_original_sample

    @torch.no_grad()
    def _step_sd3_belm(self, noise_pred, t, latents):
        latents = self.model.scheduler.step_belm_true(noise_pred, t, latents, return_dict=False)[0]
        self.latents_stack.append(latents.detach())
        return latents

    def edit(self):
        #my code
        #self.model.scheduler.set_timesteps(self.model.scheduler.num_inference_steps)
        
        latents = self.start_latent
        self.latents_stack = []
        old_pred_original_sample = None
        for i, timestep in tqdm(
                enumerate(self.model.scheduler.timesteps),
                
                #my code
                #total=self.model.scheduler.num_inference_steps,
                total=len(self.model.scheduler.timesteps),  
                          
                desc='Editing',
                disable=not self.verbose
        ):
            # 1. Construct dict 
            #careful           
            #data_dict = self._construct_data_dict(latents, i, timestep)
            data_dict = self._construct_data_dict_cogvideo(latents, i, timestep)

            # 2. Calculate guidance
            #noise_pred = self._get_noise(data_dict, i)
            noise_pred = self._get_noise_cogvideo(data_dict, i)

            # 3. Scheduler step

            #latents = self._step_cogvideo(noise_pred, timestep, latents)
            
            timestep_back = self.model.scheduler.timesteps[i - 1] if i > 0 else None
            latents, old_pred_original_sample = self._step_cogvideo_5b(noise_pred,timestep,latents,
                            old_pred_original_sample,timestep_back)
            
            #my code
            #postprocess
            
            # if i>=0 and i<0:
            #     latents = self.local_blend(latents, data_dict)
                
            if i>=0 and i<40:
                mask = self.mask.clone()
                #mask = ~self.mask.clone()
                #mask = -self.mask.clone() + 1
                
                # mask[:, [1, 2], :, :, :] = mask[:, [2, 1], :, :, :]
                # # inter = mask[:,1]
                # # mask[:,1] = mask[:,2]
                # # mask[:,2] = inter
                
                # plt.imshow(mask.squeeze().cpu()[2], cmap = "hot")
                # plt.savefig("/home/whl/workspace/cogvideo_edit/Detic/demo_test/z_swan_invert_3kernel_2frame.png")
                if i<0:
                    mask[:2,:,:] = True
                    mask[:,:2,:,:,:] = True
                #latents = data_dict["inv_latent_prev"] + self.mask * (latents - data_dict["inv_latent_prev"] ) #[1,3,16,60,90]
                latents = data_dict["inv_latent_prev"].to(latents.device) + mask.to(latents.device) * (latents - data_dict["inv_latent_prev"].to(latents.device) ) #[1,3,16,60,90]
                latents = latents.to(data_dict["inv_latent_prev"].dtype)
            

            
            
            #my code
            del data_dict
            gc.collect()
            torch.cuda.empty_cache()
            
            
            
        #self._model_unpatch(self.model)
        
        
        #return latent2video(self.model, latents, "/home/whl/workspace/cogvideo_edit/result_cogvideo/rabbit_edit_white_cat_6_3e8_2e2_4attn_bf16.mp4", 4)
        #return latent2video(self.model, latents, "/home/whl/workspace/cogvideo_edit/result_cogvideo/parrot_edit_eagle_1_3e7_2e2_4attn_bf16.mp4", 4)
        #return latent2video(self.model, latents, "/home/whl/workspace/cogvideo_edit/result_cogvideo/rabbit_resize_edit_whitecat_6_3e8_2e2_4attn_bf16.mp4", 4)
        #return latent2video(self.model, latents, "/home/whl/workspace/cogvideo_edit/result_cogvideo/rabbit_noresize_edit_whitecat_6_3e8_2e2_4attn_bf16.mp4", 4)
        #return latent2video(self.model, latents, "/home/whl/workspace/cogvideo_edit/result_cogvideo/parrot_resize_edit_plane_6_3e8_2e2_4attn_bf16_25.mp4", 4)
        
        #return latent2video(self.model, latents, "/home/whl/workspace/cogvideo_edit/result_cogvideo/cat_longtext_edit_motion_6_1e8_5e2_4attn_bf16_35_rescale3_50step.mp4", 4)
        
        #return latent2video(self.model, latents, "/home/whl/workspace/cogvideo_edit/result_cogvideo/tiger_longtext_edit_lego_6_1e8_5e2_4attn_bf16_35_rescale3_nonstyle.mp4", 4)
        
        #return latent2video(self.model, latents, "/home/whl/workspace/cogvideo_edit/result_cogvideo_new/chid_longtext_edit_panda_6_5e4_8e2_26attn_bf16_30_rescale3_diag.mp4", 4)
          
        return latent2video(self.model, latents, f"/home/cby/cogvideo_code/edit_output/gray_dog/5b_4attn_wholemask_{self.config.guiders[1].kwargs.self_attn_gs}_{self.config.guiders[1].kwargs.app_gs}_30_deticmask_3kerneldilate_40local_fft(20,30)_rescale3.mp4",
                            4,(640,640))
        
        #return latent2video(self.model, latents, "/home/whl/workspace/cogvideo_edit/result_cogvideo_motion/chid_longtext_edit_panda_fall_6_0_5e2_4result_bf16_30_rescale3.mp4", 4)
    @torch.no_grad()
    def init_prompt(self, inv_prompt: str, trg_prompt: str):
        
        #my code
        # trg_prompt_embed = self.get_prompt_embed(trg_prompt)
        # inv_prompt_embed = self.get_prompt_embed(inv_prompt)
        # uncond_embed = self.get_prompt_embed("")
        # self.context = torch.cat([uncond_embed, inv_prompt_embed, trg_prompt_embed])
        
        #sd3
        # self.uncond_embed,self.trg_prompt_embed,self.uncond_embed_pooled,self.trg_prompt_embed_pooled = self.get_prompt_embed(trg_prompt)
        # _,self.inv_prompt_embed,_,self.inv_prompt_embed_pooled = self.get_prompt_embed(inv_prompt)
        
        #cogvideo
        trg_prompt_embed = self.model._get_t5_prompt_embeds(
            prompt=trg_prompt,
            device=self.device,
            dtype=self.dtype,
        )
        inv_prompt_embed = self.model._get_t5_prompt_embeds(
            prompt=inv_prompt,
            device=self.device,
            dtype=self.dtype,
        )
        uncond_embed = self.model._get_t5_prompt_embeds(
            prompt="",
            device=self.device,
            dtype=self.dtype,
        )
        self.context = torch.cat([uncond_embed, inv_prompt_embed, trg_prompt_embed])
        

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
                    hidden_states,              #image[1024,1024]:[1,4096,1536] image[512,512]:[1,1024,1536]        
                    encoder_hidden_states=None, #[1,154,1536]
                    attention_mask=None,
                    image_rotary_emb = None
            ):
                text_seq_length = encoder_hidden_states.size(1)

                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

                query = self.to_q(hidden_states)
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)
                

                inner_dim = key.shape[-1]           #1920
                head_dim = inner_dim // self.heads  #64   attn.heads=30

                query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) #[1,30,4276,64]

                if self.norm_q is not None:
                    query = self.norm_q(query)
                if self.norm_k is not None:
                    key = self.norm_k(key)

                
                # # Apply RoPE if needed
                # if image_rotary_emb is not None:
                #     from .embeddings import apply_rotary_emb

                #     query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                #     if not attn.is_cross_attention:
                #         key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                
                
                #[1,30,4276,64]
                

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                #[1,4276,1920]

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                encoder_hidden_states, hidden_states = hidden_states.split(
                    [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
                )
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
                module._forward_hooks = OrderedDict()
            if hasattr(module, 'children'):
                for module_ in module.children():
                    remove_hooks(module_)
        #my code
        register_attn(model.transformer)
        remove_hooks(model.transformer)
