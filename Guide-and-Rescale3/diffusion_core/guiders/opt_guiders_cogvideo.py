import torch
from typing import Optional

from diffusion_core.utils.class_registry import ClassRegistry
from diffusion_core.guiders.scale_schedulers import last_steps, first_steps
import pdb
import torch.nn.functional as F
import gc
import os
import sys

sys.path.insert(0, os.path.expanduser("../cogvideo_code/Guide-and-Rescale"))
from diffusers.models.embeddings import apply_rotary_emb

opt_registry = ClassRegistry()

class BaseGuider:
    def __init__(self):
        self.clear_outputs()
    
    @property
    def grad_guider(self):
        return hasattr(self, 'grad_fn')
    
    def __call__(self, data_dict):
        if self.grad_guider:
            return self.grad_fn(data_dict)
        else:
            return self.calc_energy(data_dict)   
        
    def clear_outputs(self):
        if not self.grad_guider:
            self.output = self.single_output_clear()
    
    def single_output_clear(self):
        raise NotImplementedError()


@opt_registry.add_to_registry('cfg')
class ClassifierFreeGuidance(BaseGuider):
    def __init__(self, is_source_guidance=False):
        self.is_source_guidance = is_source_guidance
    
    def grad_fn(self, data_dict):
        prompt_unet = data_dict['src_prompt_unet'] if self.is_source_guidance else data_dict['trg_prompt_unet']
        return prompt_unet - data_dict['uncond_unet']
    
@opt_registry.add_to_registry('cfg_negative_prompt')
class ClassifierFreeGuidanceNegative(BaseGuider):
    def __init__(self, is_source_guidance=False):
        self.is_source_guidance = is_source_guidance
    
    def grad_fn(self, data_dict):
        prompt_unet = data_dict['src_prompt_unet'] if self.is_source_guidance else data_dict['trg_prompt_unet']
        return prompt_unet - data_dict['src_prompt_unet']


@opt_registry.add_to_registry('latents_diff')
class LatentsDiffGuidance(BaseGuider):
    """
    \| z_t* - z_t \|^2_2
    """
    def grad_fn(self, data_dict):
        return 2 * (data_dict['latent'] - data_dict['inv_latent'])

                
@opt_registry.add_to_registry('features_map_l2')
class FeaturesMapL2EnergyGuider(BaseGuider):
    def __init__(self, block='up'):
        assert block in ['down', 'up', 'mid', 'whole']
        self.block = block
        
    patched = True
    forward_hooks = ['cur_trg', 'inv_inv']
    def calc_energy(self, data_dict):
        return torch.mean(torch.pow(data_dict['features_map_l2_cur_trg'] - data_dict['features_map_l2_inv_inv'], 2))
    
    def model_patch(self, model, self_attn_layers_num=None):
        def hook_fn(module, input, output):
            self.output = output 
        if self.block == 'mid':
            model.unet.mid_block.register_forward_hook(hook_fn)
        elif self.block == 'up':
            model.unet.up_blocks[1].resnets[1].register_forward_hook(hook_fn)
            model.unet.up_blocks[-1].register_forward_hook(hook_fn)
        elif self.block == 'down':
            model.unet.down_blocks[1].resnets[1].register_forward_hook(hook_fn)
    
    def single_output_clear(self):
        None
    
    
@opt_registry.add_to_registry('self_attn_map_l2')
class SelfAttnMapL2EnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']    
    def single_output_clear(self):
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self":  [], "mid_self":  [], "up_self":  []
        }
    
    def calc_energy(self, data_dict):
        result = 0.
        for unet_place, data in data_dict['self_attn_map_l2_cur_inv'].items():
            for elem_idx, elem in enumerate(data):
                result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        self.single_output_clear()
        return result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_self:
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])

    
@opt_registry.add_to_registry('self_attn_map_l2_appearance')
class SelfAttnMapL2withAppearanceEnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs

    def single_output_clear(self):
        return {
            "down_self":  [], 
            "mid_self":  [], 
            "up_self":  [],
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        for unet_place in unet_places:
            data = data_dict['self_attn_map_l2_appearance_cur_inv'][unet_place]
            for elem_idx, elem in enumerate(data):
                self_attn_result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_appearance_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        
        features_orig = data_dict['self_attn_map_l2_appearance_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cur_inv']['features']
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]

        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_self:
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
                
                hidden_states = torch.bmm(attention_probs, value)

                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])
        
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.unet.up_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)

@opt_registry.add_to_registry('self_attn_map_l2_appearance_sd3')
class SelfAttnMapL2withAppearanceEnergyGuiderSD3(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs
   
    def single_output_clear(self):
        return {
            "self":  [], 
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        #my code
        # for unet_place in unet_places:
        #     data = data_dict['self_attn_map_l2_appearance_cur_inv'][unet_place]
        #     for elem_idx, elem in enumerate(data):
        #         self_attn_result += torch.mean(
        #             torch.pow(
        #                 elem - data_dict['self_attn_map_l2_appearance_inv_inv'][unet_place][elem_idx], 2
        #             )
        #         )
        data = data_dict['self_attn_map_l2_appearance_sd3_cur_inv']['self']
        for elem_idx, elem in enumerate(data):
            self_attn_result += torch.mean(
                torch.pow(
                    elem - data_dict['self_attn_map_l2_appearance_sd3_inv_inv']['self'][elem_idx], 2
                )
            )
        
        features_orig = data_dict['self_attn_map_l2_appearance_sd3_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_sd3_cur_inv']['features']
        
        #my code
        features_orig = features_orig[1]
        features_cur = features_cur[1]
        
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
        
        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self):
            def patched_forward(
                    hidden_states,              #image[1024,1024]:[1,4096,1536] image[512,512]:[1,1024,1536]        
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
                
                #my code
                #important part of attention map? 
                query_h2b = self.head_to_batch_dim(query)
                key_h2b = self.head_to_batch_dim(key)
                value_h2b = self.head_to_batch_dim(value)                
                attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)
                
                # attention_probs=attention_probs[:,:hidden_states.shape[1],:hidden_states.shape[1]]
                
                # query_h2b = self.head_to_batch_dim(query[:,:hidden_states.shape[1],:])
                # key_h2b = self.head_to_batch_dim(key[:,:hidden_states.shape[1],:])
                # value_h2b = self.head_to_batch_dim(value[:,:hidden_states.shape[1],:])                
                # attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)                
                
                guider_self.output["self"].append(attention_probs)
                

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
        
        def register_attn(module, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:

                    module.forward = new_forward_info(module)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.transformer.named_children()
        pdb.set_trace()
        for name, net in sub_nets:
            if "transformer_blocks" in name:
                register_attn(net, [0,100])
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.transformer.transformer_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)
            
@opt_registry.add_to_registry('self_attn_map_l2_appearance')
class SelfAttnMapL2withAppearanceEnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs

    def single_output_clear(self):
        return {
            "down_self":  [], 
            "mid_self":  [], 
            "up_self":  [],
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        for unet_place in unet_places:
            data = data_dict['self_attn_map_l2_appearance_cur_inv'][unet_place]
            for elem_idx, elem in enumerate(data):
                self_attn_result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_appearance_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        
        features_orig = data_dict['self_attn_map_l2_appearance_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cur_inv']['features']
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]

        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_self:
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
                
                hidden_states = torch.bmm(attention_probs, value)

                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])
        
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.unet.up_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)

@opt_registry.add_to_registry('self_attn_map_l2_appearance_cogvideo')
class SelfAttnMapL2withAppearanceEnergyGuiderCogVideo(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs
           
    def single_output_clear(self):
        
        return {
            "self":  [], 
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        data = data_dict['self_attn_map_l2_appearance_cogvideo_cur_inv']['self']
        
        for elem_idx, elem in enumerate(data):
            self_attn_result += torch.mean(
                torch.pow(
                    elem - data_dict['self_attn_map_l2_appearance_cogvideo_inv_inv']['self'][elem_idx], 2
                )
            )
        features_orig = data_dict['self_attn_map_l2_appearance_cogvideo_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cogvideo_cur_inv']['features']
        
        #my code
        features_orig = features_orig[0]
        features_cur = features_cur[0]
        
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
            

        return _self_attn_gs * self_attn_result + _app_gs * app_result    
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
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

                #my code
                query_h2b = self.head_to_batch_dim(query).float()
                key_h2b = self.head_to_batch_dim(key).float()
                value_h2b = self.head_to_batch_dim(value)
                attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)
                
                guider_self.output[f"self"].append(attention_probs)

                inner_dim = key.shape[-1]           #1920
                head_dim = inner_dim // self.heads  #64   attn.heads=30

                query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) #[1,30,4276,64]

                if self.norm_q is not None:
                    query = self.norm_q(query)
                if self.norm_k is not None:
                    key = self.norm_k(key)
                   
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                
                
                #[1,30,4276,64]
                

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                #[1,4276,1920]

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states) #[1,4276,1920]
                
                
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
                )
                          
                return hidden_states, encoder_hidden_states
            return patched_forward
        
        #my code
        def register_attn(module, layers_num, cur_layers_num=0, parent_name='', index=None):
            module_name = f'{parent_name}.{index}.{module.__class__.__name__}' if index is not None else f'{parent_name}.{module.__class__.__name__}'
            module_name = module_name.lstrip('.')  

            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    print(module_name)  
                    module.forward = new_forward_info(module)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for i, module_ in enumerate(module.children()):
                    cur_layers_num = register_attn(module_, layers_num, cur_layers_num, module_name, i)
                return cur_layers_num

            return cur_layers_num
        
        sub_nets = model.transformer.named_children()
        for name, net in sub_nets:
            if "transformer_blocks" in name:
                register_attn(net, [0,2])
        #pdb.set_trace()
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.transformer.transformer_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)

@opt_registry.add_to_registry('self_attn_map_l2_appearance_cogvideo_filter')
class SelfAttnMapL2withAppearanceEnergyGuiderCogVideoFilter(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs

           
    def single_output_clear(self):
        #my code
        return {
            "self":  [], 
            "features": None,
            "cross":[]
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        data = data_dict['self_attn_map_l2_appearance_cogvideo_filter_cur_inv']['self']
        
        for elem_idx, elem in enumerate(data):
            self_attn_result += torch.mean(
                torch.pow(
                    elem - data_dict['self_attn_map_l2_appearance_cogvideo_filter_inv_inv']['self'][elem_idx], 2
                )
            )
        features_orig = data_dict['self_attn_map_l2_appearance_cogvideo_filter_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cogvideo_filter_cur_inv']['features']
        
        #my code
        features_orig = features_orig[0]
        features_cur = features_cur[0]
        
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
            
        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
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
                
                # #my code
                # query_h2b = self.head_to_batch_dim(query).float()
                # key_h2b = self.head_to_batch_dim(key).float()
                # value_h2b = self.head_to_batch_dim(value).float()
                # attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)
                # cross_probs = attention_probs[:,-4050:,:226].clone()
                # sum_probs = cross_probs.sum(dim=-1, keepdim=True)
                # normalized_cross_probs = torch.where(sum_probs != 0, cross_probs / sum_probs, torch.zeros_like(cross_probs))                              
                # guider_self.output[f"cross"].append(normalized_cross_probs)
                
            

                inner_dim = key.shape[-1]           #1920
                head_dim = inner_dim // self.heads  #64   attn.heads=30

                query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) #[1,30,4276,64]

                if self.norm_q is not None:
                    query = self.norm_q(query)
                if self.norm_k is not None:
                    key = self.norm_k(key)                    
               
                # Apply RoPE if needed
                if image_rotary_emb is not None:

                    query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                    if not self.is_cross_attention:
                        key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
                
                # #my code    
                # query_h2b = query.reshape(-1,4276,head_dim).float()
                # key_h2b = key.reshape(-1,4276,head_dim).float()
                # attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)
                # cross_probs = attention_probs[:,-4050:,:226].clone()
                # guider_self.output[f"cross"].append(cross_probs)
                
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                
                
                #[1,30,4276,64]
                

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                #[1,4276,1920]

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states) #[1,4276,1920]
                
                
                
                # fft_tensor = torch.fft.fft2(hidden_states.to(torch.float32))
                # low_freq_tensor = fft_tensor[:, :int(4276/2), :int(1920/2)]
                # low_freq_output = torch.fft.ifft2(low_freq_tensor)
                # low_freq_output = torch.abs(low_freq_output)
                # guider_self.output[f"self"].append(low_freq_output)
                
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
                )

                tensor_input = hidden_states.reshape([hidden_states.shape[0],-1,30,45,hidden_states.shape[-1]]).to(torch.float32)
                tensor_fft = torch.fft.fft2(tensor_input, dim=(2, 3))
                tensor_fft = torch.fft.fftshift(tensor_fft, dim=(2,3))

                low_freq_size = (20, 30)
                mask = torch.zeros(tensor_fft.shape).to(tensor_fft.device)
                center_x, center_y = tensor_fft.shape[2] // 2, tensor_fft.shape[3] // 2

                x_start, x_end = center_x - low_freq_size[0] // 2, center_x + low_freq_size[0] // 2
                y_start, y_end = center_y - low_freq_size[1] // 2, center_y + low_freq_size[1] // 2
                mask[:,:,x_start:x_end, y_start:y_end, :] = 1
                tensor_fft_low_freq = tensor_fft * mask
                
                tensor_fft_low_freq = torch.fft.ifftshift(tensor_fft_low_freq, dim=(2, 3))
                tensor_output = torch.fft.ifft2(tensor_fft_low_freq, dim=(2, 3)).real
                guider_self.output[f"self"].append(tensor_output)
                              
                return hidden_states, encoder_hidden_states
            return patched_forward
        
        #my code
        def register_attn(module, layers_num, cur_layers_num=0, parent_name='', index=None):
            module_name = f'{parent_name}.{index}.{module.__class__.__name__}' if index is not None else f'{parent_name}.{module.__class__.__name__}'
            module_name = module_name.lstrip('.')  

            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    print(module_name)  
                    module.forward = new_forward_info(module)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for i, module_ in enumerate(module.children()):
                    cur_layers_num = register_attn(module_, layers_num, cur_layers_num, module_name, i)
                return cur_layers_num

            return cur_layers_num


        
        sub_nets = model.transformer.named_children()
        for name, net in sub_nets:
            if "transformer_blocks" in name:
                register_attn(net, [0,2])
        #pdb.set_trace()
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.transformer.transformer_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)
            
@opt_registry.add_to_registry('self_attn_map_l2_appearance_cogvideo_filter_new')
class SelfAttnMapL2withAppearanceEnergyGuiderCogVideoFilterNew(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs

           
    def single_output_clear(self):
        #my code
        return {
            "self":  [], 
            "features": None,
            "cross":[]
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        data = data_dict['self_attn_map_l2_appearance_cogvideo_filter_new_cur_inv']['self']
        
        for elem_idx, elem in enumerate(data):
            self_attn_result += torch.mean(
                torch.pow(
                    elem - data_dict['self_attn_map_l2_appearance_cogvideo_filter_new_inv_inv']['self'][elem_idx], 2
                )
            )
        features_orig = data_dict['self_attn_map_l2_appearance_cogvideo_filter_new_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cogvideo_filter_new_cur_inv']['features']
        
        #my code
        features_orig = features_orig[0]
        features_cur = features_cur[0]
        
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
            
        if _self_attn_gs == 0 and _self_attn_gs == 0:
            return 0*self_attn_result, 0*app_result

        return _self_attn_gs*self_attn_result, _self_attn_gs*app_result
    
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
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
                
                # #my code
                # query_h2b = self.head_to_batch_dim(query).float()
                # key_h2b = self.head_to_batch_dim(key).float()
                # value_h2b = self.head_to_batch_dim(value).float()
                # attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)
                # cross_probs = attention_probs[:,-4050:,:226].clone()
                # sum_probs = cross_probs.sum(dim=-1, keepdim=True)
                # normalized_cross_probs = torch.where(sum_probs != 0, cross_probs / sum_probs, torch.zeros_like(cross_probs))                              
                # guider_self.output[f"cross"].append(normalized_cross_probs)
                
            

                inner_dim = key.shape[-1]           #1920
                head_dim = inner_dim // self.heads  #64   attn.heads=30

                query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) #[1,30,4276,64]

                if self.norm_q is not None:
                    query = self.norm_q(query)
                if self.norm_k is not None:
                    key = self.norm_k(key)                    
               
                # Apply RoPE if needed
                if image_rotary_emb is not None:

                    query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                    if not self.is_cross_attention:
                        key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
                
                # #my code    
                # query_h2b = query.reshape(-1,4276,head_dim).float()
                # key_h2b = key.reshape(-1,4276,head_dim).float()
                # attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)
                # cross_probs = attention_probs[:,-4050:,:226].clone()
                # guider_self.output[f"cross"].append(cross_probs)
                
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                
                
                #[1,30,4276,64]
                

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
                #[1,4276,1920]

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states) #[1,4276,1920]
                
                
                
                # fft_tensor = torch.fft.fft2(hidden_states.to(torch.float32))
                # low_freq_tensor = fft_tensor[:, :int(4276/2), :int(1920/2)]
                # low_freq_output = torch.fft.ifft2(low_freq_tensor)
                # low_freq_output = torch.abs(low_freq_output)
                # guider_self.output[f"self"].append(low_freq_output)
                
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
                )

                tensor_input = hidden_states.reshape([hidden_states.shape[0],-1,30,45,hidden_states.shape[-1]]).to(torch.float32)
                tensor_fft = torch.fft.fft2(tensor_input, dim=(2, 3))
                tensor_fft = torch.fft.fftshift(tensor_fft, dim=(2,3))

                low_freq_size = (20, 30)
                mask = torch.zeros(tensor_fft.shape).to(tensor_fft.device)
                center_x, center_y = tensor_fft.shape[2] // 2, tensor_fft.shape[3] // 2

                x_start, x_end = center_x - low_freq_size[0] // 2, center_x + low_freq_size[0] // 2
                y_start, y_end = center_y - low_freq_size[1] // 2, center_y + low_freq_size[1] // 2
                mask[:,:,x_start:x_end, y_start:y_end, :] = 1
                tensor_fft_low_freq = tensor_fft * mask
                
                tensor_fft_low_freq = torch.fft.ifftshift(tensor_fft_low_freq, dim=(2, 3))
                tensor_output = torch.fft.ifft2(tensor_fft_low_freq, dim=(2, 3)).real
                guider_self.output[f"self"].append(tensor_output)
                              
                return hidden_states, encoder_hidden_states
            return patched_forward
        
        #my code
        def register_attn(module, layers_num, cur_layers_num=0, parent_name='', index=None):
            module_name = f'{parent_name}.{index}.{module.__class__.__name__}' if index is not None else f'{parent_name}.{module.__class__.__name__}'
            module_name = module_name.lstrip('.')  

            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    print(module_name)  
                    module.forward = new_forward_info(module)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for i, module_ in enumerate(module.children()):
                    cur_layers_num = register_attn(module_, layers_num, cur_layers_num, module_name, i)
                return cur_layers_num

            return cur_layers_num


        
        sub_nets = model.transformer.named_children()
        for name, net in sub_nets:
            if "transformer_blocks" in name:
                # register_attn(net, [0,2])
                register_attn(net, [0,2])
        #pdb.set_trace()
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.transformer.transformer_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)
            
@opt_registry.add_to_registry('self_attn_map_l2_appearance_cogvideo_step')
class SelfAttnMapL2withAppearanceEnergyGuiderCogVideoStep(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs
        self.step_count = 0
           
    def single_output_clear(self):
        
        return {
            "self":  [], 
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        data = data_dict['self_attn_map_l2_appearance_cogvideo_step_cur_inv']['self']
        for elem_idx, elem in enumerate(data):
            if self.step_count >= 10 and elem_idx >= 4:
                continue
            if self.step_count >= 20 and elem_idx >= 2:
                continue
            self_attn_result += torch.mean(
                torch.pow(
                    elem - data_dict['self_attn_map_l2_appearance_cogvideo_step_inv_inv']['self'][elem_idx], 2
                )
            )
        features_orig = data_dict['self_attn_map_l2_appearance_cogvideo_step_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cogvideo_step_cur_inv']['features']
        
        #my code
        features_orig = features_orig[0]
        features_cur = features_cur[0]
        
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
        self.step_count += 1
        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
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
                
                #my code
                query_h2b = self.head_to_batch_dim(query)
                key_h2b = self.head_to_batch_dim(key)
                value_h2b = self.head_to_batch_dim(value)
                attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)

                
                guider_self.output[f"self"].append(attention_probs)
                
                del query_h2b,key_h2b,value_h2b
                gc.collect()
                torch.cuda.empty_cache()

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
                
        #my code
        def register_attn(module, layers_num, cur_layers_num=0, parent_name='', index=None):
            module_name = f'{parent_name}.{index}.{module.__class__.__name__}' if index is not None else f'{parent_name}.{module.__class__.__name__}'
            module_name = module_name.lstrip('.')  

            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    print(module_name)  
                    module.forward = new_forward_info(module)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for i, module_ in enumerate(module.children()):
                    cur_layers_num = register_attn(module_, layers_num, cur_layers_num, module_name, i)
                return cur_layers_num

            return cur_layers_num


        
        sub_nets = model.transformer.named_children()
        for name, net in sub_nets:
            if "transformer_blocks" in name:
                # register_attn(net, [0,2])
                register_attn(net, [0,4])
        #pdb.set_trace()
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.transformer.transformer_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)
            
@opt_registry.add_to_registry('self_attn_map_l2_appearance')
class SelfAttnMapL2withAppearanceEnergyGuider(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs

    def single_output_clear(self):
        return {
            "down_self":  [], 
            "mid_self":  [], 
            "up_self":  [],
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        for unet_place in unet_places:
            data = data_dict['self_attn_map_l2_appearance_cur_inv'][unet_place]
            for elem_idx, elem in enumerate(data):
                self_attn_result += torch.mean(
                    torch.pow(
                        elem - data_dict['self_attn_map_l2_appearance_inv_inv'][unet_place][elem_idx], 2
                    )
                )
        
        features_orig = data_dict['self_attn_map_l2_appearance_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cur_inv']['features']
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]

        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
        def new_forward_info(self, place_unet):
            def patched_forward(
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if self.spatial_norm is not None:
                    hidden_states = self.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                if self.group_norm is not None:
                    hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = self.to_q(hidden_states)
                
                ## Injection
                is_self = encoder_hidden_states is None
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif self.norm_cross:
                    encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)
                if is_self:
                    guider_self.output[f"{place_unet}_self"].append(attention_probs)
                
                hidden_states = torch.bmm(attention_probs, value)

                hidden_states = self.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = self.to_out[0](hidden_states)
                # dropout
                hidden_states = self.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / self.rescale_output_factor

                return hidden_states
            return patched_forward
        
        def register_attn(module, place_in_unet, layers_num, cur_layers_num=0):
            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    module.forward = new_forward_info(module, place_in_unet)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for module_ in module.children():
                    cur_layers_num = register_attn(module_, place_in_unet, layers_num, cur_layers_num)
                return cur_layers_num
        
        sub_nets = model.unet.named_children()
        for name, net in sub_nets:
            if "down" in name:
                register_attn(net, "down", self_attn_layers_num[0])
            if "mid" in name:
                register_attn(net, "mid", self_attn_layers_num[1])
            if "up" in name:
                register_attn(net, "up", self_attn_layers_num[2])
        
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.unet.up_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)

@opt_registry.add_to_registry('self_attn_map_l2_appearance_diag')
class SelfAttnMapL2withAppearanceEnergyGuiderDiag(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs
   
           
    def single_output_clear(self):
        
        return {
            "self":  [], 
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        data = data_dict['self_attn_map_l2_appearance_diag_cur_inv']['self']
        for elem_idx, elem in enumerate(data):
            self_attn_result += torch.mean(torch.pow(
                    elem- data_dict['self_attn_map_l2_appearance_diag_inv_inv']['self'][elem_idx], 2
                ))
        features_orig = data_dict['self_attn_map_l2_appearance_diag_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_diag_cur_inv']['features']
        
        #my code
        features_orig = features_orig[0]
        features_cur = features_cur[0]
        
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
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
                
                #my code
                query_h2b = self.head_to_batch_dim(query)
                key_h2b = self.head_to_batch_dim(key)
                value_h2b = self.head_to_batch_dim(value)
                attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)
                guider_self.output[f"self"].append(attention_probs[:,:,:226].clone())
                del query_h2b,key_h2b,value_h2b
                gc.collect()
                torch.cuda.empty_cache()

                inner_dim = key.shape[-1]           #1920
                head_dim = inner_dim // self.heads  #64   attn.heads=30

                query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) #[1,30,4276,64]

                if self.norm_q is not None:
                    query = self.norm_q(query)
                if self.norm_k is not None:
                    key = self.norm_k(key)

                
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
               
        
        #my code
        def register_attn(module, layers_num, cur_layers_num=0, parent_name='', index=None):
            module_name = f'{parent_name}.{index}.{module.__class__.__name__}' if index is not None else f'{parent_name}.{module.__class__.__name__}'
            module_name = module_name.lstrip('.') 

            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    print(module_name) 
                    module.forward = new_forward_info(module)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for i, module_ in enumerate(module.children()):
                    cur_layers_num = register_attn(module_, layers_num, cur_layers_num, module_name, i)
                return cur_layers_num

            return cur_layers_num


        
        sub_nets = model.transformer.named_children()
        for name, net in sub_nets:
            if "transformer_blocks" in name:
                # register_attn(net, [0,2])
                register_attn(net, [0,13])
        #pdb.set_trace()
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.transformer.transformer_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)
            
            
@opt_registry.add_to_registry('self_attn_map_l2_appearance_cogvideo_motion')
class SelfAttnMapL2withAppearanceEnergyGuiderCogVideoMotion(BaseGuider):
    patched = True
    forward_hooks = ['cur_inv', 'inv_inv']

    def __init__(
        self, self_attn_gs: float, app_gs: float, new_features: bool=False, 
        total_last_steps: Optional[int] = None, total_first_steps: Optional[int] = None
    ):
        super().__init__()
        
        self.new_features = new_features

        if total_last_steps is not None:
            self.app_gs = last_steps(app_gs, total_last_steps)
            self.self_attn_gs = last_steps(self_attn_gs, total_last_steps)
        elif total_first_steps is not None:
            self.app_gs = first_steps(app_gs, total_first_steps)
            self.self_attn_gs = first_steps(self_attn_gs, total_first_steps)
        else:
            self.app_gs = app_gs
            self.self_attn_gs = self_attn_gs
   
    # #my code
    # def clear_outputs(self):

    #     if not self.grad_guider:
    #         if hasattr(self, 'output'):
    #             self.output.clear()
    #             gc.collect()
    #             torch.cuda.empty_cache()         
    #         self.output =  {
    #         "self":  [], 
    #         "features": None
    #     }
           
    def single_output_clear(self):
        #my code
        # return {
        #     "down_self":  [], 
        #     "mid_self":  [], 
        #     "up_self":  [],
        #     "features": None
        # }
        
        return {
            "self":  [], 
            "features": None
        }
    
    def calc_energy(self, data_dict):
        self_attn_result = 0.
        unet_places = ['down_self', 'up_self', 'mid_self']
        #my code
        # for unet_place in unet_places:
        #     data = data_dict['self_attn_map_l2_appearance_cur_inv'][unet_place]
        #     for elem_idx, elem in enumerate(data):
        #         self_attn_result += torch.mean(
        #             torch.pow(
        #                 elem - data_dict['self_attn_map_l2_appearance_inv_inv'][unet_place][elem_idx], 2
        #             )
        #         )
        data = data_dict['self_attn_map_l2_appearance_cogvideo_motion_cur_inv']['self']

        for elem_idx, elem in enumerate(data):
            self_attn_result += torch.mean(
                torch.pow(
                    elem - data_dict['self_attn_map_l2_appearance_cogvideo_motion_inv_inv']['self'][elem_idx], 2
                )
            )
        # print(f"cur_inv {data_dict['self_attn_map_l2_appearance_cogvideo_cur_inv']['self'][2].flatten()[:20]}")
        # print(f"inv_inv {data_dict['self_attn_map_l2_appearance_cogvideo_inv_inv']['self'][2].flatten()[:20]}")
        features_orig = data_dict['self_attn_map_l2_appearance_cogvideo_motion_inv_inv']['features']
        features_cur = data_dict['self_attn_map_l2_appearance_cogvideo_motion_cur_inv']['features']
        
        #my code
        features_orig = features_orig[0]
        features_cur = features_cur[0]
        
        app_result = torch.mean(torch.abs(features_cur - features_orig))

        self.single_output_clear()

        if type(self.app_gs) == float:
            _app_gs = self.app_gs
        else:
            _app_gs = self.app_gs[data_dict['diff_iter']]

        if type(self.self_attn_gs) == float:
            _self_attn_gs = self.self_attn_gs
        else:
            _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
        return _self_attn_gs * self_attn_result + _app_gs * app_result
    
    # def calc_energy(self, data_dict):
    #     self_attn_result = 0.
    #     unet_places = ['down_self', 'up_self', 'mid_self']
    #     #my code
    #     # for unet_place in unet_places:
    #     #     data = data_dict['self_attn_map_l2_appearance_cur_inv'][unet_place]
    #     #     for elem_idx, elem in enumerate(data):
    #     #         self_attn_result += torch.mean(
    #     #             torch.pow(
    #     #                 elem - data_dict['self_attn_map_l2_appearance_inv_inv'][unet_place][elem_idx], 2
    #     #             )
    #     #         )
    #     data = data_dict['self_attn_map_l2_appearance_cogvideo_cur_inv']['self']
    #     for elem_idx, elem in enumerate(data):
    #         self_attn_result += torch.mean(
    #             torch.pow(
    #                 elem.cpu().to(torch.float32) - data_dict['self_attn_map_l2_appearance_cogvideo_inv_inv']['self'][elem_idx].cpu().to(torch.float32), 2
    #             )
    #         )
        
    #     features_orig = data_dict['self_attn_map_l2_appearance_cogvideo_inv_inv']['features']
    #     features_cur = data_dict['self_attn_map_l2_appearance_cogvideo_cur_inv']['features']
        
    #     #my code
    #     features_orig = features_orig[0].cpu().to(torch.float32)
    #     features_cur = features_cur[0].cpu().to(torch.float32)
        
    #     app_result = torch.mean(torch.abs(features_cur - features_orig))

    #     self.single_output_clear()

    #     if type(self.app_gs) == float:
    #         _app_gs = self.app_gs
    #     else:
    #         _app_gs = self.app_gs[data_dict['diff_iter']]

    #     if type(self.self_attn_gs) == float:
    #         _self_attn_gs = self.self_attn_gs
    #     else:
    #         _self_attn_gs = self.self_attn_gs[data_dict['diff_iter']]
    #     return (_self_attn_gs * self_attn_result + _app_gs * app_result).to(torch.bfloat16)
    
    def model_patch(guider_self, model, self_attn_layers_num=None):
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
                
                # #my code
                # query_h2b = self.head_to_batch_dim(query)
                # key_h2b = self.head_to_batch_dim(key)
                # value_h2b = self.head_to_batch_dim(value)
                # attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)
                # guider_self.output[f"self"].append(attention_probs)
                # del query_h2b,key_h2b,value_h2b
                # gc.collect()
                # torch.cuda.empty_cache()

                inner_dim = key.shape[-1]           #1920
                head_dim = inner_dim // self.heads  #64   attn.heads=30

                query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) #[1,30,4276,64]

                if self.norm_q is not None:
                    query = self.norm_q(query)
                if self.norm_k is not None:
                    key = self.norm_k(key)
                

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
                
                #my code
                #guider_self.output[f"self"].append(hidden_states)
                
                guider_output = hidden_states[:,:-text_seq_length,:]
                guider_output = guider_output.reshape([batch_size,-1,30,45,hidden_states.shape[-1]])
                
                guider_self.output[f"self"].append(guider_output[:,0,:,:,:])
                

                encoder_hidden_states, hidden_states = hidden_states.split(
                    [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
                )
                return hidden_states, encoder_hidden_states
            return patched_forward
        
        #my code
        def register_attn(module, layers_num, cur_layers_num=0, parent_name='', index=None):
            # 构建当前模块的完整名称，如果有索引则加入索引
            module_name = f'{parent_name}.{index}.{module.__class__.__name__}' if index is not None else f'{parent_name}.{module.__class__.__name__}'
            module_name = module_name.lstrip('.')  # 去掉前导的点号

            if 'Attention' in module.__class__.__name__:
                if 2 * layers_num[0] <= cur_layers_num < 2 * layers_num[1]:
                    print(module_name)  # 打印完整路径名称，包括索引
                    module.forward = new_forward_info(module)
                return cur_layers_num + 1
            elif hasattr(module, 'children'):
                for i, module_ in enumerate(module.children()):
                    # 递归调用时传递当前模块的完整名称和索引作为父级名称
                    cur_layers_num = register_attn(module_, layers_num, cur_layers_num, module_name, i)
                return cur_layers_num

            return cur_layers_num


        
        sub_nets = model.transformer.named_children()
        for name, net in sub_nets:
            if "transformer_blocks" in name:
                # register_attn(net, [0,2])
                register_attn(net, [0,2])
        #pdb.set_trace()
        def hook_fn(module, input, output):
            guider_self.output["features"] = output

        if guider_self.new_features:
            model.transformer.transformer_blocks[-1].register_forward_hook(hook_fn)
        else:
            model.unet.conv_norm_out.register_forward_hook(hook_fn)
