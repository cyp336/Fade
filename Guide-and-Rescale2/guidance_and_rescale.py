import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf


#from diffusion_core.guiders.guidance_editing_sd3 import GuidanceEditing
from diffusion_core.guiders.guidance_editing import GuidanceEditing

from diffusion_core.utils import load_512, use_deterministic
from diffusion_core import diffusion_models_registry, diffusion_schedulers_registry
import os
import pdb
import sys


current_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_directory)
from diffusers import StableDiffusion3Pipeline,StableDiffusionPipeline,DDIMScheduler



os.environ["CUDA_VISIBLE_DEVICES"]="7"
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
    #important
    model.to(torch.bfloat16)
    return model

scheduler_name = 'ddim_50_eps'
scheduler = get_scheduler(scheduler_name)

model_name = 'stable-diffusion-v1-4'
model = get_model(scheduler, model_name, device)

# model = StableDiffusion3Pipeline.from_pretrained("/home/whl/workspace/sd3/model_sd3", 
#                                                 torch_dtype=torch.float32,
#                                                 text_encoder_3=None,
#                                                 tokenizer_3=None,)



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

image_path = "/home/whl/workspace/sd3/Guide-and-Rescale/example_images/zebra.jpeg"
image_path = "/home/whl/workspace/sd3/Guide-and-Rescale/result/bird.jpg"
image = Image.fromarray(load_512(image_path))
# init_prompt = "A photo of a zebra"
# edit_prompt = "A photo of a tiger"
init_prompt = "A bird is flying in the forest"
edit_prompt = "A eagle is flying in the forest"
#config = OmegaConf.load('/home/whl/workspace/sd3/Guide-and-Rescale/configs/ours_nonstyle_best.yaml')
config = OmegaConf.load('/home/whl/workspace/sd3/Guide-and-Rescale/configs/ours_nonstyle_best_colab.yaml')
guidance = GuidanceEditing(model, config)
res = guidance(image, init_prompt, edit_prompt, verbose=True)
show_images(np.asarray(image), res, init_prompt, edit_prompt,'/home/whl/workspace/sd3/Guide-and-Rescale/result_ddim/eagle.jpg')
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

