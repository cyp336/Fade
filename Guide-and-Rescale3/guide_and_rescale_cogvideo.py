import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import imageio
from torchvision import transforms

from diffusion_core.guiders.guidance_editing_cogvideo import GuidanceEditing
#from diffusion_core.guiders.guidance_editing_cogvideo_new import GuidanceEditing
#from diffusion_core.guiders.guidance_editing import GuidanceEditing

from diffusion_core.utils import load_512, use_deterministic
from diffusion_core import diffusion_models_registry, diffusion_schedulers_registry
import os
import pdb
import sys

sys.path.insert(0, os.path.expanduser("../"))
import diffusers
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils import export_to_video
import torch.nn.functional as F

import ptp_utils
import random

class LocalBlend:

    # def __call__(self, x_t, attention_store):
    #     #x_t [2,4,64,64]
    #     k = 1
    #     maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3] #len 5 shape [16,256,77] 
    #     #self.alpha_layers [2,1,1,1,1,77]
    #     maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, 226) for item in maps] #len 5 shape [2,8,1,16,16,77]
    #     maps = torch.cat(maps, dim=1) #[2,40,1,16,16,77]
    #     maps = (maps * self.alpha_layers).sum(-1).mean(1) #[2,1,16,16]
    #     mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k)) #[2,1,16,16]
    #     mask = F.interpolate(mask, size=(x_t.shape[2:])) #[2,1,64,64]
    #     mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0] #[2,1,64,64]
    #     mask = mask.gt(self.threshold)
    #     mask = (mask[:1] + mask[1:]).float() #[1,1,64,64]
    #     x_t = x_t[:1] + mask * (x_t - x_t[:1])
    #     return x_t
    
    # def __call__(self, x_t, data_dict):

    #     k = 1
    #     maps_inv = data_dict["self_attn_map_l2_appearance_cogvideo_inv_inv"]["self"]
    #     maps_cur = data_dict["self_attn_map_l2_appearance_cogvideo_cur_inv"]["self"]        
        
    #     maps_inv = [item[:,-4050:,:226].float() for item in maps_inv] #[30,4050,226] * 4
    #     maps_cur = [item[:,-4050:,:226].float() for item in maps_cur] #[30,4050,226] * 4
        
    #     maps_inv = [item.reshape(1, -1, 3, 30, 45, 226) for item in maps_inv] #[1,30,3,30,45,226] * 4
    #     maps_cur = [item.reshape(1, -1, 3, 30, 45, 226) for item in maps_cur] #[1,30,3,30,45,226] * 4
        
    #     maps_inv = torch.cat(maps_inv, dim = 1) #[1,30*4,3,30,45,226]
    #     maps_cur = torch.cat(maps_cur, dim = 1) #[1,30*4,3,30,45,226]
        
    #     maps = torch.cat([maps_inv,maps_cur],dim = 0) #[2,30*4,3,30,45,226]
    #     maps = (maps * self.alpha_layers).sum(-1).mean(1) #[2,3,30,45]
    #     mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k)) #[2,3,30,45]
    #     mask = mask.unsqueeze(2) #[2,3,1,30,45]
    #     mask = F.interpolate(mask, size=(1, 60, 90), mode='trilinear', align_corners=False) #[2,3,1,60,90]
    #     mask = mask / mask.max(3, keepdims=True)[0].max(4, keepdims=True)[0] #[2,3,1,60,90]
    #     mask = mask.gt(self.threshold)
    #     mask = (mask[:1] + mask[1:]).float() #[1,3,1,60,90]
    #     if data_dict["diff_iter"] == 28:
    #         torch.save(mask,"/home/whl/workspace/cogvideo_edit/result_cogvideo_latest/mask_28.pt")
    #     x_t = data_dict["inv_latent_prev"] + mask * (x_t - data_dict["inv_latent_prev"] ) #[1,3,16,60,90]
    #     #x_t = data_dict["inv_latent"] + mask * (x_t - data_dict["inv_latent"] ) #[1,3,16,60,90]
    #     x_t = x_t.to(data_dict["inv_latent"].dtype)
        
    #     return x_t
    
    #all block all head
    def __call__(self, x_t, data_dict):

        k = 1
        maps_inv = data_dict["self_attn_map_l2_appearance_cogvideo_filter_inv_inv"]["cross"]
        maps_cur = data_dict["self_attn_map_l2_appearance_cogvideo_filter_cur_inv"]["cross"]        
        
        maps_inv = [item.float() for item in maps_inv] #[30,4050,226] * 4
        maps_cur = [item.float() for item in maps_cur] #[30,4050,226] * 4
        
        maps_inv = [item.reshape(1, -1, 3, 30, 45, 226) for item in maps_inv] #[1,30,3,30,45,226] * 4
        maps_cur = [item.reshape(1, -1, 3, 30, 45, 226) for item in maps_cur] #[1,30,3,30,45,226] * 4
        
        maps_inv = torch.cat(maps_inv, dim = 1) #[1,30*4,3,30,45,226]
        maps_cur = torch.cat(maps_cur, dim = 1) #[1,30*4,3,30,45,226]
        
        maps = torch.cat([maps_inv,maps_cur],dim = 0) #[2,30*4,3,30,45,226]
        maps = (maps * self.alpha_layers).sum(-1).mean(1) #[2,3,30,45]
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k)) #[2,3,30,45]
        mask = mask.unsqueeze(2) #[2,3,1,30,45]
        mask = F.interpolate(mask, size=(1, 60, 90), mode='trilinear', align_corners=False) #[2,3,1,60,90]
        mask = mask / mask.max(3, keepdims=True)[0].max(4, keepdims=True)[0] #[2,3,1,60,90]
        
        # if data_dict["diff_iter"] == 28:
        #     torch.save(mask,"")        
        
        mask = mask.gt(self.threshold)
        
        
        mask = (mask[:1] + mask[1:]).float() #[1,3,1,60,90]
        x_t = data_dict["inv_latent_prev"] + mask * (x_t - data_dict["inv_latent_prev"] ) #[1,3,16,60,90]
        #x_t = data_dict["inv_latent"] + mask * (x_t - data_dict["inv_latent"] ) #[1,3,16,60,90]
        x_t = x_t.to(data_dict["inv_latent"].dtype)
        
        return x_t
    
    # #4block 8head
    # def __call__(self, x_t, data_dict):

    #     k = 1
    #     maps_inv = data_dict["self_attn_map_l2_appearance_cogvideo_filter_inv_inv"]["cross"]
    #     maps_cur = data_dict["self_attn_map_l2_appearance_cogvideo_filter_cur_inv"]["cross"]        
        
    #     maps_inv = [item.float() for item in maps_inv] #[30,4050,226] * 4
    #     maps_cur = [item.float() for item in maps_cur] #[30,4050,226] * 4
        
    #     maps_inv = [item.reshape(1, -1, 3, 30, 45, 226) for item in maps_inv] #[1,30,3,30,45,226] * 4
    #     maps_cur = [item.reshape(1, -1, 3, 30, 45, 226) for item in maps_cur] #[1,30,3,30,45,226] * 4
        
    #     maps_inv = maps_inv[4][:,8,:,:,:,:]
    #     maps_cur = maps_cur[4][:,8,:,:,:,:]

    #     # maps_inv = torch.cat(maps_inv, dim = 1) #[1,30*4,3,30,45,226]
    #     # maps_cur = torch.cat(maps_cur, dim = 1) #[1,30*4,3,30,45,226]
        
    #     maps = torch.cat([maps_inv,maps_cur],dim = 0) #[2,30*4,3,30,45,226]
    #     maps = (maps * self.alpha_layers).sum(-1).mean(1) #[2,3,30,45]
    #     mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k)) #[2,3,30,45]
    #     mask = mask.unsqueeze(2) #[2,3,1,30,45]
    #     mask = F.interpolate(mask, size=(1, 60, 90), mode='trilinear', align_corners=False) #[2,3,1,60,90]
    #     mask = mask / mask.max(3, keepdims=True)[0].max(4, keepdims=True)[0] #[2,3,1,60,90]
        
    #     if data_dict["diff_iter"] == 28:
    #         torch.save(mask,"/home/whl/workspace/cogvideo_edit/result_cogvideo_filter_latest/mask_28_4attn_8head.pt")
        
    #     mask = mask.gt(self.threshold)
    #     mask = (mask[:1] + mask[1:]).float() #[1,3,1,60,90]
    #     x_t = data_dict["inv_latent_prev"] + mask * (x_t - data_dict["inv_latent_prev"] ) #[1,3,16,60,90]
    #     #x_t = data_dict["inv_latent"] + mask * (x_t - data_dict["inv_latent"] ) #[1,3,16,60,90]
    #     x_t = x_t.to(data_dict["inv_latent"].dtype)
        
    #     return x_t
       
    def __init__(self, prompts, words, tokenizer, threshold=.3, device="cuda"):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, 226)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                #ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                
                alpha_layers[i, :, :, :, :, ind] = 1
        #my code important
        alpha_layers[1:] = alpha_layers[:1]        
                
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold
        
# class LocalBlend:
    
#     def get_mask(self, x_t, maps, alpha, use_pool):
#         k = 1
#         maps = (maps * alpha).sum(-1).mean(2)
#         if use_pool:
#             maps = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
#         mask = F.interpolate(maps, size=(x_t.shape[3:]))
#         mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
#         mask = mask.gt(self.th[1-int(use_pool)])
#         mask = mask[:1] + mask
#         return mask
    
#     def __call__(self, x_t, attention_store, step):
#         self.counter += 1
#         if self.counter > self.start_blend:
#             maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
#             maps = [item.reshape(self.alpha_layers.shape[0], -1, 8, 16, 16, 226) for item in maps]
#             maps = torch.cat(maps, dim=2)
#             mask = self.get_mask(x_t, maps, self.alpha_layers, True)
#             if self.substruct_layers is not None:
#                 maps_sub = ~self.get_mask(x_t, maps, self.substruct_layers, False)
#                 mask = mask * maps_sub
#             mask = mask.float()
#             mask = mask.reshape(-1, 1, mask.shape[-3], mask.shape[-2], mask.shape[-1])
#             x_t = x_t[:1] + mask * (x_t - x_t[:1])
#         return x_t
    
#     def __init__(self, prompts, words,tokenizer,substruct_words=None, start_blend=0.2, th=(.3, .3),device="cuda"):
#         alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, 226)
#         for i, (prompt, words_) in enumerate(zip(prompts, words)):
#             if type(words_) is str:
#                 words_ = [words_]
#             for word in words_:
#                 ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
#                 alpha_layers[i, :, :, :, :, ind] = 1
        
#         if substruct_words is not None:
#             substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, 226)
#             for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
#                 if type(words_) is str:
#                     words_ = [words_]
#                 for word in words_:
#                     ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
#                     substruct_layers[i, :, :, :, :, ind] = 1
#             self.substruct_layers = substruct_layers.to(device)
#         else:
#             self.substruct_layers = None
#         self.alpha_layers = alpha_layers.to(device)
#         NUM_DDIM_STEPS = 50
#         self.start_blend = int(start_blend * NUM_DDIM_STEPS)
#         self.counter = 0 
#         self.th=th
    



def get_scheduler(scheduler_name):
    if scheduler_name not in diffusion_schedulers_registry:
        raise ValueError(f"Incorrect scheduler type: {scheduler_name}, possible are {diffusion_schedulers_registry}")
    scheduler = diffusion_schedulers_registry[scheduler_name]()
    return scheduler

def get_model(scheduler, model_name, device):
    model = diffusion_models_registry[model_name](scheduler)
    model.to(device)
    return model

def set_sigmas(pipe,num_inference_steps=50,device="cuda", timesteps=None):
    import numpy as np
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = pipe.transformer.config.in_channels
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def show_images(init_image, edit_image, init_prompt, edit_prompt,save_path):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    axs[0].imshow(init_image)
    axs[0].axis('off')
    axs[0].set_title('Initial image\nPrompt: \"{}\"'.format(init_prompt), fontsize=10)
    
    axs[1].imshow(edit_image)
    axs[1].axis('off')
    axs[1].set_title('Edited image\nPrompt: \"{}\"'.format(edit_prompt), fontsize=10)
    
    plt.show()
    plt.savefig(save_path)

def latent2video(pipe,latent,output_path,fps=4):
    
    latent = latent.permute(0,2,1,3,4)
    latent = latent / pipe.vae.config.scaling_factor
    with torch.no_grad():
        decoded_frames = pipe.vae.decode(latent).sample
        
    frames = decoded_frames[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)

    # writer = imageio.get_writer(output_path + "/output.mp4", fps=24)
    writer = imageio.get_writer(output_path, fps=fps)

    for frame in frames:
        writer.append_data(frame)
    writer.close()

    return decoded_frames

def video2latent(pipe, video_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and encodes the video frames.

    Parameters:
    - pipe: CogVideoXPipeline.
    - video_path (str): The path to the video file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The encoded video frames.
    """
    video_reader = imageio.get_reader(video_path, "ffmpeg")
    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    if len(frames)>49:
        step=5
        new_frames=frames[0:step*29:step]
        frames=new_frames
    video_reader.close()
    
    resize = transforms.Resize((480, 720))
    frames=[resize(frame) for frame in frames]
    
    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)
    #frames_tensor = torch.stack(frames).to(device).unsqueeze(0).to(dtype)
    # frames_tensor = frames_tensor[:, :, :97, :256, :256]
    with torch.no_grad():
        encoded_frames = pipe.vae.encode(frames_tensor)[0].sample()
        #print(f'encoded_frames = {encoded_frames}')
        
    latent=encoded_frames.permute(0,2,1,3,4) #[batch_size, num_frames=k+1,num_channels=16, h=H/8, w=W/8] [1,8,16,60,90]
    latent=pipe.vae.config.scaling_factor * latent
    
    return latent

def encode_multi_image(pipe, image_path, dtype, device):
    import os
    from PIL import Image
    import torch
    from torchvision import transforms

    #jpg_files = [os.path.join(image_path, f) for f in sorted(os.listdir(image_path)) if f.endswith('.jpg')]
    image_files = [os.path.join(image_path, f) for f in sorted(os.listdir(image_path)) if f.endswith('.jpg')]
    frames = []

    resize = transforms.Resize((480, 720))
    transform = transforms.ToTensor()

    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        image=resize(image)
        image_tensor = transform(image)
        frames.append(image_tensor)
    
    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)
    print(f'frames_tensor.shape = {frames_tensor.shape}')
    with torch.no_grad():
        encoded_frames = pipe.vae.encode(frames_tensor)[0].sample()
    
    return encoded_frames

    


def decode_video(pipe, latent, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and decodes the encoded video frames.

    Parameters:
    - pipe: CogVideoXPipeline.
    -latent
    #- encoded_tensor_path (str): The path to the encoded tensor file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The decoded video frames.
    """
    #encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)
    encoded_frames=latent
    with torch.no_grad():
        print(f'encoded_frames.shape = {encoded_frames.shape}')
        decoded_frames = pipe.vae.decode(encoded_frames).sample
    return decoded_frames

def save_video(tensor, output_path,fps):
    """
    Saves the video frames to a video file.

    Parameters:
    - tensor (torch.Tensor): The video frames tensor.
    - output_path (str): The path to save the output video.
    """
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)

    # writer = imageio.get_writer(output_path + "/output.mp4", fps=24)
    writer = imageio.get_writer(output_path, fps=fps)

    for frame in frames:
        writer.append_data(frame)
    writer.close()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="4,5"
    
    use_deterministic()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16
    num_inference_steps = 50

    # #load the cogvideo pipe
    # pipe = CogVideoXPipeline.from_pretrained("/mnt/nfs/CogVideoX-2b",torch_dtype=dtype)
    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    
    # #load the cogvideo-5b
    # pipe = CogVideoXPipeline.from_pretrained("/mnt/nfs/CogVideoX-5b",torch_dtype=dtype)
    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()  
      
    #load the cogvideo-5b 
    pipe = CogVideoXPipeline.from_pretrained("/mnt/nfs/CogVideoX-5b/", torch_dtype=dtype,device_map="balanced")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    
    #inference
    # video = pipe(
    #     prompt="A playful cat wearing a harness is sitting on the grass beneath a blossoming cherry tree, surrounded by vibrant pink petals. The cat turns its head curiously, scanning the environment with bright, alert eyes, as if noticing something in the distance. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The sunlight filters through the blossoms, casting soft shadows on the cat's fur, while a slight breeze causes the petals to flutter around. The background features a park setting with faint outlines of structures blurred, adding depth to the scene.",
    #     num_videos_per_prompt=1,  # Number of videos to generate per prompt
    #     num_inference_steps=num_inference_steps,  # Number of inference steps
    #     # num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.31.0` and after.
    #     num_frames=9,
    #     use_dynamic_cfg=False,  ## This id used for DPM Sechduler, for DDIM scheduler, it should be False
    #     guidance_scale=6,  # Guidance scale for classifier-free guidance, can set to 7 for DPM scheduler
    #     generator=torch.Generator().manual_seed(0),  # Set the seed for reproducibility
    # ).frames[0]
    # export_to_video(video, "/home/whl/workspace/cogvideo_edit/pigeon_direct.mp4", fps=4)
    
    
    # video_path = ""
    # latent_video = video2latent(pipe,video_path,dtype=dtype,device=device)
    
    # init_prompt = "A rabbit jump on the grass"
    # edit_prompt = "A white cat jump on the grass"
    
    # init_prompt = "A parrot is flying in the forest"
    # edit_prompt = "A plane is flying in the forest"
    
    # init_prompt = "A cat is sitting outdoors under a blossoming tree"
    # edit_prompt = "A dog is sitting outdoors under a blossoming tree"
    
    #init_prompt = "A playful cat wearing a harness is sitting on the grass beneath a blossoming cherry tree, surrounded by vibrant pink petals. The cat turns its head curiously, scanning the environment with bright, alert eyes, as if noticing something in the distance. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The sunlight filters through the blossoms, casting soft shadows on the cat's fur, while a slight breeze causes the petals to flutter around. The background features a park setting with faint outlines of structures blurred, adding depth to the scene."
    #edit_prompt = "A playful dog wearing a harness is sitting on the grass beneath a blossoming cherry tree, surrounded by vibrant pink petals. The dog turns its head curiously, scanning the environment with bright, alert eyes, as if noticing something in the distance. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The sunlight filters through the blossoms, casting soft shadows on the dog's fur, while a slight breeze causes the petals to flutter around. The background features a park setting with faint outlines of structures blurred, adding depth to the scene."
    #edit_prompt = "A playful tiger wearing a harness is sitting on the grass beneath a blossoming cherry tree, surrounded by vibrant pink petals. The tiger turns its head curiously, scanning the environment with bright, alert eyes, as if noticing something in the distance. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The sunlight filters through the blossoms, casting soft shadows on the tiger's fur, while a slight breeze causes the petals to flutter around. The background features a park setting with faint outlines of structures blurred, adding depth to the scene."
    #edit_prompt = "A young tiger sits on the grassy ground beneath a canopy of blooming cherry blossoms, surrounded by scattered pink petals. The tiger's head turns attentively, its keen eyes scanning the area as if drawn by a distant sound. Its graceful movements reflect the natural flow of the scene, while the cherry tree’s branches sway gently overhead. Sunlight filters through the blossoms, casting a delicate play of light and shadows across the tiger's striped fur. A light breeze stirs the petals around, creating a dynamic, serene atmosphere. The park setting in the background features blurred outlines of structures, enhancing the sense of depth."
    
    #init_prompt = "A cat, dressed in a black harness with subtle patterns, sits comfortably on the grass under the blossoming cherry tree. It starts by looking to the left. Gradually, the cat turns its head to the right, pausing halfway as if something catches its eye, and then continues the motion, its gaze scanning the surroundings with curiosity. At one point, the cat briefly opens its mouth as if to meow or catch a scent in the air. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The background features a park setting with faint outlines of structures blurred, adding depth to the scene."
    #edit_prompt = "A dog, dressed in a black harness with subtle patterns, sits comfortably on the grass under the blossoming cherry tree. It starts by looking to the left. Gradually, the dog turns its head to the right, pausing halfway as if something catches its eye, and then continues the motion, its gaze scanning the surroundings with curiosity. At one point, the dog briefly opens its mouth as if to meow or catch a scent in the air. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The background features a park setting with faint outlines of structures blurred, adding depth to the scene."
    #edit_prompt = "A tiger, dressed in a black harness with subtle patterns, sits comfortably on the grass under the blossoming cherry tree. It starts by looking to the left. Gradually, the tiger turns its head to the right, pausing halfway as if something catches its eye, and then continues the motion, its gaze scanning the surroundings with curiosity. At one point, the tiger briefly opens its mouth as if to meow or catch a scent in the air. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The background features a park setting with faint outlines of structures blurred, adding depth to the scene."
    #edit_prompt = "A tiger sits comfortably on the grass under the blossoming cherry tree. It starts by looking to the left. Gradually, the tiger turns its head to the right, pausing halfway as if something catches its eye, and then continues the motion, its gaze scanning the surroundings with curiosity. At one point, the tiger briefly opens its mouth as if to meow or catch a scent in the air. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The background features a park setting with faint outlines of structures blurred, adding depth to the scene."
    #edit_prompt = "A cat, adorned with a stylish collar featuring delicate floral designs, lounges peacefully on the grass beneath a vibrant cherry blossom tree. It begins by stretching its front paws out in front, arching its back in a slow, elegant movement. After the stretch, the cat turns its body to the right, settling back down and curling into a comfortable position, its tail flicking playfully. It then raises its head slightly, eyes wide and alert, as if noticing a fluttering butterfly nearby. The cat’s movements are smooth and relaxed, reflecting the gentle swaying of the tree’s branches. The background showcases a tranquil park, with softly blurred shapes of benches and pathways, adding a sense of serenity to the scene."
    
    #init_prompt = "A sequence capturing an owl perched on a gloved hand, preparing to take flight in a serene outdoor setting. The background is softly blurred, highlighting the owl's motion as it gradually unfolds its wings. Warm lighting from the setting or rising sun illuminates the scene, casting a gentle glow on the owl's feathers. The video showcases the elegance and strength of the owl, with its wings spreading wider frame by frame, transitioning from a calm stance to taking flight. The setting is tranquil, emphasizing the natural grace and beauty of the bird in motion."
    #edit_prompt = "A sequence capturing a pigeon perched on a gloved hand, preparing to take flight in a serene outdoor setting. The background is softly blurred, highlighting the pigeon's motion as it gradually unfolds its wings. Warm lighting from the setting or rising sun illuminates the scene, casting a gentle glow on the pigeon's feathers. The video showcases the elegance and strength of the pigeon, with its wings spreading wider frame by frame, transitioning from a calm stance to taking flight. The setting is tranquil, emphasizing the natural grace and beauty of the bird in motion."    
    #edit_prompt = "A sequence capturing a robotic owl perched on a gloved hand, preparing to take flight in a serene outdoor setting. The background is softly blurred, emphasizing the owl's mechanical movements as it gradually unfolds its metallic wings. Warm lighting from the setting or rising sun reflects off the owl's sleek, polished surface, casting a subtle glow on its intricately designed metal feathers. The video showcases the blend of technology and grace, with the owl's wings spreading wider frame by frame, transitioning from a stationary pose to taking flight. The setting is tranquil, enhancing the contrast between the natural environment and the robotic bird in motion."
    
    #init_prompt = "A young child wearing a teal shirt, black shorts, and a white safety helmet rides a small balance bike along a curved sidewalk in a suburban park. The child grips the handlebars confidently, focusing ahead while learning to maintain balance. The grassy surroundings add a touch of greenery, creating a relaxed outdoor setting. The video captures the slight forward movement, displaying determination and a sense of adventure. The background includes gently sloping lawns and a paved path, emphasizing the everyday charm of a neighborhood stroll while learning to ride."
    #edit_prompt = "A panda rides a small balance bike along a curved sidewalk in a suburban park. The panda grips the handlebars with its fluffy paws, focusing ahead while trying to maintain balance on the bike. Its furry face shows determination and a hint of playful curiosity. The grassy surroundings add a natural touch, creating a charming outdoor setting. Each frame captures the panda’s slight forward movement, highlighting its adventurous spirit. The background includes gently sloping lawns and a paved path, enhancing the playful and whimsical nature."
    #edit_prompt = "A young child wearing a teal shirt, black shorts, and a white safety helmet rides a small balance bike along a gently winding sidewalk in a suburban park. The child occasionally glances to the side, captivated by a fluttering butterfly, which causes them to momentarily wobble but quickly regain balance. The grassy surroundings provide a lush backdrop, enhancing the cheerful outdoor atmosphere. The video highlights this brief moment of distraction, capturing both the child's determination and a sense of wonder. The background features softly sloping lawns and a paved path, showcasing the charming simplicity of a neighborhood adventure while learning to ride."
    #edit_prompt = "A panda rides a small balance bike along a winding path in a suburban park, its fluffy paws gripping the handlebars with determination. As it focuses ahead, a sudden miscalculation causes the panda to wobble, and with a surprised look, it loses balance. In a comical moment, the panda tumbles gently onto the soft grass beside the path, its round face a mix of surprise and amusement. The grassy surroundings provide a soft landing, while vibrant flowers sway nearby, enhancing the lighthearted atmosphere. Each frame captures the panda's playful spirit, turning a fall into a delightful adventure as it shakes off the tumble and prepares to try again."
    
    
    #init_prompt = "A majestic tiger prowls through a lush forest, its powerful body moving gracefully across the grassy terrain. The sunlight filters through the canopy, casting dappled shadows on the tiger's vibrant orange and black-striped coat. Each frame captures the fluidity of its motion as it steps cautiously, its keen eyes scanning the surroundings. The tiger's posture is both alert and relaxed, embodying a perfect blend of strength and elegance. In the background, tall trees and scattered branches add to the natural ambiance, highlighting the wild beauty of the setting. The scene conveys a sense of quiet power and the timeless allure of the jungle."
    #edit_prompt = "A tiger in LEGO style prowls through a lush forest, its blocky, plastic form contrasting with the realistic grassy terrain. The sunlight filters through the canopy, casting shadows on the tiger's orange and black-striped LEGO bricks, giving it a playful yet lifelike appearance. Each frame captures the robotic yet graceful motion of the LEGO tiger as it steps forward, its articulated joints mimicking the movements of a real animal. The tiger’s head turns slightly, with bright, plastic eyes scanning the surroundings. The background features natural trees and scattered branches, creating a blend of realism and LEGO creativity in the tranquil forest setting."
    
    # init_prompt = "A silver jeep driving along a winding mountain road lined with greenery and tall pine trees. It starts from a distance, with the jeep approaching and coming gradually closer in each frame. As it rounds the bend, the car's details become more visible, including its compact, rugged shape. In the background, tall, rocky mountains add a dramatic backdrop, while the bright daylight enhances the vivid colors of the trees and road. It captures a feeling of exploration and adventure as the jeep progresses along the scenic path."
    # edit_prompt = "A red Porsche driving along a winding mountain road lined with greenery and tall pine trees. It starts from a distance, with the Porsche approaching and gradually coming closer in each frame. As it rounds the bend, the car's details becomes more visible, including its compact, rugged shape. In the background, tall, rocky mountains add a dramatic backdrop, while the bright daylight enhances the vivid colors of the trees and road. It captures a feeling of exploration and adventure as the Porsche progresses along the scenic path."
    
    #init_prompt = "A graceful black swan glides serenely through a calm pond, its sleek feathers and vibrant red beak creating a striking contrast against the green water and lush foliage. The swan moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the swan's posture and movement, portraying a peaceful moment in nature as the swan explores its pond with calm curiosity."
    #edit_prompt = "A graceful white duck glides serenely through a calm pond, its soft feathers and bright orange beak creating a striking contrast against the green water and lush foliage. The duck moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the duck's posture and movement, portraying a peaceful moment in nature as the duck explores its pond with calm curiosity."
    #edit_prompt = "A graceful white duck glides serenely through a calm pond, its white feathers and bright orange beak creating a striking contrast against the green water and lush foliage. The white duck moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the duck's posture and movement, portraying a peaceful moment in nature as the duck explores its pond with calm curiosity."
    #edit_prompt = "A pure white duck glides serenely through a calm pond, its pure white feathers and bright orange beak creating a striking contrast against the green water and lush foliage. The pure white duck moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the duck's posture and movement, portraying a peaceful moment in nature as the duck explores its pond with calm curiosity."
    #edit_prompt = "A pure yellow duck glides serenely through a calm pond, its pure yellow feathers and bright orange beak creating a striking contrast against the green water and lush foliage. The pure yellow duck moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the duck's posture and movement, portraying a peaceful moment in nature as the duck explores its pond with calm curiosity."
    #edit_prompt = "An graceful icy swan glides serenely through a calm pond, its frosty feathers and glistening blue beak creating a striking contrast against the green water and lush foliage. The swan moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the swan's posture and movement, portraying a peaceful moment in nature as the swan explores its pond with calm curiosity."
    #edit_prompt = "An ice swan glides gracefully across a tranquil, frost-kissed pond, its body appearing as though sculpted from crystalline ice with a sleek, frosty sheen. Its feathers, delicate and translucent, shimmer with a cool blue glow, refracting soft glints of light with each subtle movement. The swan’s neck arches elegantly, trailing a gentle mist that swirls in the crisp air around it. Its beak glows a deep, icy blue, contrasting against the intricate facets of its frozen plumage. As the swan moves, delicate frost particles scatter from its wings, leaving an ethereal trail on the glassy surface of the pond. The surrounding icy vegetation reflects in the frozen waters, amplifying the serene, otherworldly beauty of this peaceful winter scene."
    #edit_prompt = "A graceful green duck glides serenely through a calm pond, its green feathers and bright orange beak creating a striking contrast against the green water and lush foliage. The green duck moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the duck's posture and movement, portraying a peaceful moment in nature as the duck explores its pond with calm curiosity."
    #edit_prompt = "A pure green duck glides serenely through a calm pond, its pure green feathers and bright orange beak creating a striking contrast against the green water and lush foliage. The pure green duck moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the duck's posture and movement, portraying a peaceful moment in nature as the duck explores its pond with calm curiosity."
    #edit_prompt = "A vibrant yellow duck with perfectly smooth, rounded contours glides serenely across a calm pond. Its pure, sunlit yellow feathers and bright, almost glossy orange beak create a stunning visual contrast against the deep green water and lush, leafy foliage. The duck’s compact, oval body and gently curved neck add to its soft, graceful presence, moving with quiet elegance. As it dips its bright yellow head and rounded beak into the water, gentle ripples spread outward, enhancing the serene beauty of the scene. The rich, green vegetation reflects softly in the pond, emphasizing the duck's striking yellow hue. Each frame captures subtle shifts in the duck’s rounded form and gentle movements, portraying a peaceful, idyllic moment in nature as the duck explores its pond with calm curiosity."
    #edit_prompt = "A crystal swan glides gracefully through an enchanted twilight pond, its body formed from shimmering, translucent crystals that catch and refract the fading light, casting iridescent rainbows with each elegant movement. The swan's feathers resemble delicate, geometric shards, shifting in hues from frosty blues to soft lavenders and ethereal pinks, making it appear as a living jewel. As it moves, gentle beams of light scatter from its crystalline edges, creating intricate patterns that dance across the surface of the water. The backdrop features luminous, mystical foliage that softly glows in the dusky air, while wisps of mist swirl delicately around the swan, enhancing the magical atmosphere. The scene is tranquil and serene, with twinkling motes of light floating through the air, adding to the otherworldly charm of this enchanted setting."
    #edit_prompt = "A futuristic digital swan glides elegantly across a neon-lit virtual pond, its body composed of sleek, holographic panels that shift in color between iridescent whites, soft blues, and warm purples. The swan’s “feathers” are pixelated, shimmering with a delicate glow that reflects off the surrounding digital water. Its semi-transparent wings are made of thin, luminescent screens that ripple gracefully with every movement, casting faint trails of light as it swims. A subtle array of holographic particles follows in its wake, creating a glistening path. The background is a dark, simulated pond with glowing, abstract lily pads and light particles drifting around, pulsing gently in sync with the swan's movements. The overall scene is both peaceful and futuristic, exuding a calm, ethereal beauty."
    #edit_prompt = "A graceful black swan glides serenely through a calm pond, its sleek feathers and vibrant red beak creating a striking contrast against the green water and lush foliage. The swan moves with quiet elegance, occasionally unfurling its long wings in a majestic display, revealing a breathtaking span of dark, glossy feathers that shimmer in the sunlight. Each gentle movement creates soft ripples across the water's surface, while the surrounding vegetation, rich and green, reflects in the pond, adding to the tranquil atmosphere. Every subtle shift in the swan's posture showcases its beauty as it explores its peaceful environment with calm curiosity."
    #edit_prompt = "A black swan suddenly lifts its wings, spreading them wide in a graceful arch that reveals the full span of its sleek, dark feathers, each one glistening like polished obsidian. As it unfurls its wings, the movement is both powerful and elegant, sending ripples cascading across the still pond. The sun catches the delicate edges of each feather, creating a mesmerizing play of light that dances across the water's surface, enhancing the serene beauty of the scene. In that moment, the swan embodies both majesty and tranquility, the lush green foliage framing the display, creating a striking contrast to the vivid black of its wings."
    #edit_prompt = "A graceful black swan glides slowly through the still pond, its head beginning to rise in a gentle, unhurried motion. The curve of its long neck gradually unfurls, lifting with quiet elegance as tiny droplets slip from its feathers, creating soft ripples that dance across the water’s surface. Its red beak, previously close to the water, now points forward, steady and intent. The swan’s dark eyes take in its surroundings, exuding a calm curiosity as it reaches the peak of this upward motion. The subtle grace of the lift transforms the scene, highlighting the swan's poised beauty amid the tranquil waters"
    #edit_prompt = "A graceful black swan glides serenely through a calm pond set in an autumn landscape, its sleek black feathers and vibrant red beak creating a striking contrast against the rich, warm reflections in the water. The surrounding trees and foliage are dressed in shades of gold, amber, and rust, casting dappled reflections across the pond. As the swan moves with quiet elegance, it occasionally dips its head, leaving gentle ripples in its wake. Each frame captures the soft transitions in the swan's graceful posture, highlighting a tranquil autumn moment as it explores its pond with calm curiosity."
    #edit_prompt = "A graceful black swan glides serenely through a tranquil pond set in a winter landscape. The water is partially frozen, with delicate ice patterns forming along the edges, while soft snow blankets the ground, creating a serene white canvas. Nearby, tall evergreen trees stand stoically, their branches heavy with glistening snow, reflecting the pale winter sun. As the swan paddles gracefully, small ripples break the surface, momentarily disturbing the perfect reflection of the frosty scenery, embodying the quiet beauty of a winter's day."
    #edit_prompt = "A graceful black swan glides serenely through a calm pond set in an autumn landscape. The swan's sleek black feathers and vivid red beak stand out against the warm reflections on the water, where fallen leaves float gently on the surface. Around the pond, golden, amber, and rust-colored trees create a rich, vibrant background. The stone cottages in the distance are covered in red and orange ivy, with a hint of smoke rising from chimneys, adding to the cozy, tranquil atmosphere. The peaceful scene captures the quiet elegance of the swan as it explores the lake, surrounded by the beauty of a crisp autumn day."
    #edit_prompt = "A charming, cartoon-style black swan glides playfully through a calm, animated pond, its sleek, exaggerated feathers and bright red, oversized beak adding a whimsical touch against the vibrant green water and stylized, lush foliage. The swan moves with a lighthearted elegance, occasionally dipping its round head and curved neck into the water, creating gentle, rippling waves that add to the cartoonish effect. The surrounding plants, rendered in rich greens with simple, rounded shapes, reflect in the water, enhancing the tranquil yet playful atmosphere of this cartoon scene. Each frame captures exaggerated shifts in the swan's posture and expressions, depicting a peaceful, lighthearted moment as the swan explores its animated pond with curious charm."
    #edit_prompt = "A graceful black swan glides serenely through a calm pond, captured in the rich, textured brushstrokes of an oil painting. Its sleek, dark feathers are rendered with deep layers of color, and its vibrant red beak stands out as a bold accent against the earthy greens and golden highlights of the water. The swan moves with quiet elegance, each stroke conveying the subtle play of light across its form. Surrounding foliage, lush and dense, is painted in varying shades of green, blending softly into the water's reflection with painterly depth. Gentle ripples spread in intricate arcs around the swan's path, adding a sense of tranquil movement. Each moment is captured with an artist's touch, as subtle shifts in the swan's posture and its interaction with the water come alive, portraying a timeless, serene moment in nature that evokes a sense of calm beauty."
    #edit_prompt = "A graceful black swan glides serenely through a calm pond, its textured feathers and vibrant red beak painted in rich, layered strokes that stand out against the deep green water and lush, impressionistic foliage. The swan moves with quiet elegance, occasionally dipping its head and neck into the water, leaving soft, rippling brushstrokes in its wake. The surrounding vegetation, rich with earthy greens, reflects subtly in the water, enhancing the tranquil, timeless quality of the scene. Each detail captures shifts in the swan's posture and movement, evoking a peaceful moment in nature through the classic beauty of oil painting."
    
    #init_prompt = "A brown bear walking slowly along a rocky enclosure in a zoo-like setting. The bear's thick fur has a rich, earthy tone that contrasts against the stone walls and rugged rocks surrounding it. Each frame captures a slight change in the bear's posture as it moves forward, suggesting a calm, natural pace. Sunlight filters through, casting soft shadows and highlighting the textures of the bear's fur and the surrounding rocks. The scene has a serene, almost contemplative quality, focusing on the bear's graceful movements and the peaceful atmosphere of its enclosure."
    #edit_prompt = "A panda walking slowly along a rocky enclosure in a zoo-like setting. The panda's black-and-white fur stands out against the stone walls and rugged rocks around it, creating a striking contrast. Each frame shows a subtle shift in the panda's posture as it moves forward at a calm, deliberate pace. Sunlight filters through, casting gentle shadows and illuminating the textures of the panda's fur and the surrounding rocks. The scene feels serene and tranquil, emphasizing the panda's gentle movements and the peaceful atmosphere of its naturalistic enclosure."
    
    #init_prompt = "A black swan with a red beak swimming in a river near a wall and bushes."
    #edit_prompt = "A white duck with a red beak swimming in a river near a wall and bushes."
    
    #init_prompt = "A child, wearing a white helmet, rides a small red balance bike along a paved sidewalk surrounded by grass. The child, with a focused expression, grips the handlebars as he carefully balances on the bike without pedals. Each frame captures the child's steady progression, his young face showing determination and curiosity. Sunlight softly illuminates the scene, creating a playful and heartwarming atmosphere."
    #edit_prompt = "A panda, with black-and-white fur, rides a small red balance bike along a paved sidewalk surrounded by grass. The panda, with a focused expression, grips the handlebars as it carefully balances on the bike without pedals. Each frame captures the panda's steady progression, its fluffy face showing determination and curiosity. Sunlight softly illuminates the scene, creating a playful and heartwarming atmosphere."
    
    #init_prompt = "A man is kiteboarding across a vibrant turquoise sea on a sunny day. Wearing a black helmet and colorful shorts, he skillfully maneuvers the board over the water's surface, creating a trail of white spray behind him. His arms are extended as he grips the kite's handle, harnessing the wind to glide forward with ease. The background features distant hills and a faint coastline, adding depth to the scene. The video captures the dynamic motion of kiteboarding, with the man's movements fluid and controlled against the stunning backdrop of open water and clear blue skies."
    #edit_prompt = "Spider-Man is kiteboarding across a vibrant turquoise sea on a sunny day. Dressed in his iconic red and blue suit, he skillfully maneuvers the board over the water’s surface, leaving a trail of white spray behind him. His arms are extended as he grips the kite's handle, using his super agility to harness the wind and glide forward with ease. The background features distant hills and a faint coastline, adding depth to the scene. The video captures the dynamic, action-packed motion of Spider-Man kiteboarding, his movements fluid and controlled against the stunning backdrop of open water and clear blue skies."
    #edit_prompt = "Iron Man is kiteboarding across a vibrant turquoise sea on a sunny day. Wearing his iconic red and gold suit and a black helmet, he skillfully maneuvers the board over the water's surface, creating a trail of white spray behind him. His arms are extended as he grips the kite's handle, harnessing the wind to glide forward with ease. The background features distant hills and a faint coastline, adding depth to the scene. The video captures the dynamic motion of kiteboarding, with Iron Man's movements fluid and controlled against the stunning backdrop of open water and clear blue skies."
    
    #init_prompt = "A man is surfing on a white wave"
    #edit_prompt = "Iron Man is surfing on a white wave"
    
    #init_prompt = "A small, adorable squirrel sits comfortably on a soft yellow cushion, holding a large carrot with both tiny paws. The squirrel nibbles at the carrot, its fluffy tail and pointy ears adding to its endearing appearance. Its little hands clutch the carrot firmly, bringing it closer as it takes delicate bites. The background shows a cozy indoor setting, creating a warm, homely atmosphere. The squirrel's expressions reveal focus and enjoyment as it savors the snack, with its movements gentle and deliberate. The scene is heartwarming and showcases the charm of this tiny creature indulging in its treat."
    #edit_prompt = "A small, futuristic robot mouse sits on a soft yellow cushion, holding a large carrot with its metallic paws. The robot’s design is sleek and cute, with tiny gears and LED eyes that give it an expressive appearance. Its mechanical hands grip the carrot, bringing it closer as it performs a gentle, nibbling motion, mimicking a real mouse’s eating behavior. The background shows a cozy indoor setting, enhancing the contrast between the soft surroundings and the robot's metallic frame. The robot mouse’s movements are precise and smooth, creating a charming scene as it “enjoys” its snack in an unexpectedly endearing way."
    
    #init_prompt = "A black-and-white rabbit with sleek fur sits beside a juicy slice of watermelon on a wooden surface. The rabbit curiously leans toward the vibrant red fruit, sniffing and nibbling delicately at its edge. Its long, upright ears twitch slightly, adding to its lively demeanor. The simple indoor setting features a neutral background, emphasizing the playful interaction between the rabbit and the watermelon. The rabbit's gentle movements and the vivid contrast of the fruit's colors create a charming and refreshing scene. The moment captures the rabbit's natural curiosity and the simple joy of enjoying a sweet summer treat."
    #edit_prompt = "A sleek orange tabby cat with glossy fur sits beside a juicy slice of watermelon on a wooden surface. The cat curiously leans toward the vibrant red fruit, sniffing and gently pawing at its edge. Its sharp, alert ears flick slightly, adding to its lively demeanor. The simple indoor setting features a neutral background, emphasizing the playful interaction between the cat and the watermelon. The cat's graceful movements and the vivid contrast of the fruit's colors create a charming and refreshing scene. The moment captures the cat's natural curiosity and the simple joy of exploring a sweet summer treat."
    
    #init_prompt = "A man is kiteboarding across a vibrant turquoise sea on a sunny day. Wearing a black helmet and colorful shorts, he skillfully maneuvers the board over the water's surface, creating a trail of white spray behind him. His arms are extended as he grips the kite's handle, harnessing the wind to glide forward with ease. The background features distant hills and a faint coastline, adding depth to the scene. The video captures the dynamic motion of kiteboarding, with the man's movements fluid and controlled against the stunning backdrop of open water and clear blue skies."
    #edit_prompt = "A superhero resembling Spider-Man, dressed in his iconic red-and-blue suit, is captured in a painting-style scene, performing a daring balancing act on a long wooden plank laid across a set of stairs. The artwork portrays Spider-Man with impressionistic brushstrokes, emphasizing the texture of his suit and the intricate web patterns on it. The setting is a bustling urban environment, rendered in soft, muted tones that create a sense of movement and flow. Spider-Man's arms are outstretched for balance, exuding both agility and concentration. In the background, the steel railing and scattered trees are depicted with delicate detail, adding depth and a hint of realism. This captivating moment combines elements of heroism and precision, drawing the viewer into the scene with its artistic interpretation of a precarious yet skillful act."
    
    init_prompt = "A cheerful black-and-white Border Collie walks enthusiastically alongside its owner in a quiet suburban neighborhood. The dog, full of energy, looks back at the camera with an alert and happy expression. The man, dressed casually in a navy shirt and jeans, strolls calmly, holding the leash firmly. The scene captures a moment of companionship, framed by a sunny day with a clear blue sky and a house surrounded by lush greenery in the background. The dog's lively demeanor contrasts with the owner's relaxed pace, creating a heartwarming snapshot of their walk."
    edit_prompt = "A cheerful black-and-white Border Collie walks enthusiastically alongside its owner in a quiet suburban neighborhood. The dog, full of energy, repeatedly lowers its head, tilting it slightly as it curiously inspects the ground, its expression alert and focused. With each step, the dog's head lowers again, almost as if tracking something of interest on the path. The man, dressed casually in a navy shirt and jeans, strolls calmly, holding the leash firmly. The scene captures a moment of companionship, framed by a sunny day with a clear blue sky and a house surrounded by lush greenery in the background."
    config = OmegaConf.load("./configs/ours_nonstyle_best_cogvideo.yaml")
    
    #lb = LocalBlend([init_prompt,edit_prompt], ("jeep", "Porsche"),tokenizer = pipe.tokenizer)
    
    # for self_attn_gs in range(4000,12000,1000):
    #     for app_gs in range(0,45,5):
    #         config.guiders[1].kwargs.self_attn_gs = self_attn_gs
    #         config.guiders[1].kwargs.app_gs = app_gs
    #         guidance = GuidanceEditing(pipe, config,lb=None)
    #         res = guidance(latent_video.cpu(), init_prompt, edit_prompt, verbose=True)
    
    guidance = GuidanceEditing(pipe, config,lb=None)
    res = guidance(None, init_prompt, edit_prompt, verbose=True)

if __name__=="__main__":
    main()
