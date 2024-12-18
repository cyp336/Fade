import torch
import os
import sys
import pdb 
current_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_directory)
from diffusers import StableDiffusion3Pipeline,StableDiffusionPipeline,DDIMScheduler
from diffusers.image_processor import VaeImageProcessor

def transfer_latent_to_image(pipe,latent):
    latent=latent.clone()
    latent = (latent / pipe.vae.config.scaling_factor)
    with torch.no_grad():
        image = pipe.vae.decode(latent, return_dict=False)[0]
    
    image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(pipe.vae.config.block_out_channels) - 1))
    image = image_processor.postprocess(image, output_type="pil")
    
    return image[0]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_ckpt = "../model_sd14"
prompt = "a bird flying in the forest"
device="cuda"

all_latents=torch.load("/home/whl/workspace/sd3/Guide-and-Rescale/result/bird_ddim_trajectory_sd14.pt")

pipe = StableDiffusionPipeline.from_pretrained(model_ckpt,torch_dtype=torch.float32)
pipe=pipe.to(device)


image = transfer_latent_to_image(pipe,all_latents[0])
image.save("/home/whl/workspace/sd3/Guide-and-Rescale/result/bird_ddim_decode.jpg")  


image = pipe(
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=torch.Generator().manual_seed(2024),
    latents=all_latents[-1]
).images[0]
image.save("/home/whl/workspace/sd3/Guide-and-Rescale/result/bird_ddim_recon.jpg")  