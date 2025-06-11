import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf


from diffusion_core.guiders.guidance_editing_belm import GuidanceEditing
#from diffusion_core.guiders.guidance_editing import GuidanceEditing

from diffusion_core.utils import load_512, use_deterministic
from diffusion_core import diffusion_models_registry, diffusion_schedulers_registry
import os
import pdb
import sys


current_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_directory)
from diffusers import StableDiffusion3Pipeline,StableDiffusionPipeline,DDIMScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps



os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_deterministic()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    
# scheduler_name = 'ddim_50_eps'
# scheduler = get_scheduler(scheduler_name)

# model_name = 'stable-diffusion-v1-4'
# model = get_model(scheduler, model_name, device)

model = StableDiffusion3Pipeline.from_pretrained("/home/whl/workspace/sd3/model_sd3", 
                                                torch_dtype=torch.float32,
                                                text_encoder_3=None,
                                                tokenizer_3=None,)
model.enable_model_cpu_offload() 
set_sigmas(model,num_inference_steps=50)

image_path = "/home/whl/workspace/sd3/Guide-and-Rescale/example_images/zebra.jpeg"
image = Image.fromarray(load_512(image_path))
init_prompt = "A photo of a zebra"
edit_prompt = "A photo of a white horse"
config = OmegaConf.load('/home/whl/workspace/sd3/Guide-and-Rescale/configs/ours_nonstyle_best_sd3.yaml')
guidance = GuidanceEditing(model, config)
res = guidance(image, init_prompt, edit_prompt, verbose=True)
show_images(np.asarray(image), res, init_prompt, edit_prompt,'/home/whl/workspace/sd3/Guide-and-Rescale/result_belm/horse_0_0_cfg1_2_change5.jpg')
pdb.set_trace()
    
image_path = "/home/whl/workspace/sd3/Guide-and-Rescale/example_images/face.png"
image = Image.fromarray(load_512(image_path))
init_prompt = "A photo"
edit_prompt = "Anime style face"
config = OmegaConf.load('/home/whl/workspace/sd3/Guide-and-Rescale/configs/ours_style_best.yaml')
guidance = GuidanceEditing(model, config)
res = guidance(image, init_prompt, edit_prompt, verbose=True)
show_images(np.asarray(image), res, init_prompt, edit_prompt,'/home/whl/workspace/sd3/Guide-and-Rescale/result/girl.jpg')
pdb.set_trace()
