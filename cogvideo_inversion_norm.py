import torch
import numpy as np
from diffusers.utils import export_to_video,load_image
import imageio
from torchvision import transforms
from typing import Union
from tqdm.auto import tqdm
import math
import os
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb 

sys.path.insert(0, os.path.expanduser('/home/whl/workspace/fatezero'))
import diffusers
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.models.embeddings import apply_rotary_emb


output_cross = []
step_count = 0

def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                #out.append(i+1)
                out.append(i)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

class LocalBlend:
    def __init__(self, prompts, words, tokenizer, threshold=.2, device="cuda"):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, 226)
        
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
    
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold

    def __call__(self, cross_map):

        k = 1
        maps = cross_map   
        
        if type(maps) == list:
            maps= [item[-30:].float() for item in maps]
        else:
            maps= maps[-30:].float()
        maps = [item.reshape(1, -1, 3, 30, 45, 226) for item in maps] #[1,30,3,30,45,226] * 4
        
        # only use the last one
        maps = maps[-1]
        #maps = torch.cat(maps, dim = 1) #[1,30*4,3,30,45,226]
        
        maps = (maps * self.alpha_layers).sum(-1).mean(1) #[1,3,30,45]
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k)) #[1,3,30,45]
        mask = maps
        mask = mask.unsqueeze(2) #[1,3,1,30,45]
        mask = F.interpolate(mask, size=(1, 60, 90), mode='trilinear', align_corners=False) #[1,3,1,60,90]
        mask = mask / mask.max(3, keepdims=True)[0].max(4, keepdims=True)[0] #[1,3,1,60,90]
        return mask

def model_patch(model, self_attn_layers_num=None):
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

            # Apply RoPE if needed
            if image_rotary_emb is not None:
                query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                if not self.is_cross_attention:
                    key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
                    
            #my code
            global output_cross,step_count
            query_h2b = query.reshape(-1,4276,head_dim).float()
            key_h2b = key.reshape(-1,4276,head_dim).float()
            attention_probs = self.get_attention_scores(query_h2b, key_h2b, attention_mask)
            if step_count < 4 :
                #output_cross = attention_probs[:,-4050:,:226]
                output_cross.append(attention_probs[:,-4050:,:226])
            step_count+=1

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
            register_attn(net, [13,15])
            
@torch.no_grad()
def ddim_loop(pipe, latent,num_inference_steps,prompt_embeds,negative_prompt_embeds):
    # uncond_embeddings, cond_embeddings = pipe.context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    dtype = latent.dtype
    for i in tqdm(range(num_inference_steps)):
        # if i==num_inference_steps-1:
        #     continue
        t = pipe.scheduler.timesteps[len(pipe.scheduler.timesteps) - i - 1] 
        t = t.expand(latent.shape[0])
        with torch.no_grad():
            #my code 
            #be careful
            #noise_pred = pipe.transformer(hidden_states=latent, timestep=t, encoder_hidden_states=prompt_embeds)[0].float() 
            image_rotary_emb = (
                pipe._prepare_rotary_positional_embeddings(480, 720, 3, latent.device)
                if pipe.transformer.config.use_rotary_positional_embeddings
                else None
            )            
            noise_pred = pipe.transformer(
                    hidden_states=latent,
                    encoder_hidden_states=prompt_embeds,
                    timestep=t,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0].float()
        latent = next_step(pipe,noise_pred, t, latent)
        latent = latent.to(dtype)
        all_latent.append(latent)
    return all_latent

@torch.no_grad()
def next_step(pipe, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
    timestep, next_timestep = min(timestep -pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps, 999), timestep
    
    pipe.scheduler.alphas_cumprod=pipe.scheduler.alphas_cumprod.to('cuda:0') #note: cuda:0 refer to the device
    
    alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep] if timestep >= 0 else pipe.scheduler.final_alpha_cumprod
    alpha_prod_t_next = pipe.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    
    #model_output is v predicted, transfer v prediction to noise prediction
    alpha_prod_t_next = alpha_prod_t_next.to(sample.device)
    beta_prod_t = beta_prod_t.to(sample.device)
    alpha_prod_t = alpha_prod_t.to(sample.device)
    
    model_output=(1-alpha_prod_t_next)**0.5*sample+alpha_prod_t_next**0.5*model_output 
    
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample.to(torch.float16) #note torch.float16 refer to the dtype

## ddim inversion with classifer-guidance free
@torch.no_grad()
def invert(pipe,start_latents, prompt_embeds,negative_prompt_embeds,guidance_scale=1, num_inference_steps=50,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device='cuda'):
  

    # latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []
    
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)
    pipe.scheduler.alphas_cumprod=pipe.scheduler.alphas_cumprod.to('cuda')
    
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps-1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1: continue

        t = timesteps[i]

        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        current_t = max(0, t.item() - (1000//num_inference_steps))#t
        next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]
        t=t.expand(latent_model_input.shape[0])
        #t=t.expand(1)
        
        # predict the noise residual
        noise_pred = pipe.transformer(hidden_states=latent_model_input, encoder_hidden_states=prompt_embeds,
                                      timestep=t,return_dict=False,)[0].float()

        #model_output is v predicted, transfer v prediction to noise prediction
        noise_pred=(1-alpha_t_next)**0.5*latent_model_input+alpha_t_next**0.5*noise_pred        

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
        latents=latents.to(torch.float16)
        # Store
        intermediate_latents.append(latents)            
    return torch.cat(intermediate_latents)

def encode_video(pipe, video_path, dtype, device):
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
    video_reader.close()

    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)
    #frames_tensor = torch.stack(frames).to(device).unsqueeze(0).to(dtype)
    # frames_tensor = frames_tensor[:, :, :97, :256, :256]
    print(f'frames_tensor.shape = {frames_tensor.shape}')
    with torch.no_grad():
        encoded_frames = pipe.vae.encode(frames_tensor)[0].sample()
        #print(f'encoded_frames = {encoded_frames}')
    return encoded_frames

def encode_multi_image(pipe, image_path, dtype, device):
    import os
    from PIL import Image
    import torch
    from torchvision import transforms

    image_files = [os.path.join(image_path, f) for f in sorted(os.listdir(image_path)) if f.endswith('.png') or f.endswith('.jpg')]
    frames = []

    transform = transforms.ToTensor()
    resize = transforms.Resize((480, 720))

    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        image = resize(image)
        image_tensor = transform(image)
        frames.append(image_tensor)

    #important 
    if len(frames) == 8:
        frames.append(frames[-1])
    else:
        frames = frames[:9]
    
    #important
    frames = [frames[1]]*9
    
    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)
    with torch.no_grad():
        encoded_frames = pipe.vae.encode(frames_tensor)[0].sample()
    
    return encoded_frames

    


def decode_video(pipe, latent, dtype, device):

    #encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)
    encoded_frames=latent
    with torch.no_grad():
        decoded_frames = pipe.vae.decode(encoded_frames).sample

    return decoded_frames

def save_video(tensor, output_path,fps):

    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).to(torch.float32).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)

    writer = imageio.get_writer(output_path, fps=fps)

    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
def generate_video_with_latent(pipe,prompt,output_path,num_inference_steps,latent,guidance_scale,num_frames,fps):
    
    video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            # num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.31.0` and after.
            num_frames=num_frames,
            use_dynamic_cfg=False,  ## This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance, can set to 7 for DPM scheduler
            #generator=torch.Generator().manual_seed(2024),  # Set the seed for reproducibility
            #latents=latent,
            latents=latent
        ).frames[0]
    export_to_video(video, output_path, fps=fps)

@torch.no_grad()
def transfer_image_to_latent(pipe,image_path,video_save_path,dtype=torch.bfloat16,device="cuda",fps=4):
    video_encoded=encode_multi_image(pipe,image_path,dtype,device) 
    video_encode_decode=decode_video(pipe,video_encoded,dtype,device)
    save_video(video_encode_decode,video_save_path,fps)
    
    latent=video_encoded.permute(0,2,1,3,4)
    latent=pipe.vae.config.scaling_factor * latent
    
    return latent

@torch.no_grad()
def transfer_video_to_latent(pipe,video_path,video_save_path,dtype=torch.bfloat16,device="cuda",fps=4):
    video_encoded=encode_video(pipe,video_path,dtype,device) 
    video_encode_decode=decode_video(pipe,video_encoded,dtype,device)
    save_video(video_encode_decode,video_save_path,fps)
    
    latent=video_encoded.permute(0,2,1,3,4)
    latent=pipe.vae.config.scaling_factor * latent
    
    return latent

@torch.no_grad()
def transfer_latent_to_video(pipe,latent,video_save_path,dtype=torch.bfloat16,device="cuda",fps=4):
    
    latent=latent.permute(0,2,1,3,4)
    latent=1 / pipe.vae.config.scaling_factor * latent
    
    video_encode_decode=decode_video(pipe,latent,dtype,device)
    save_video(video_encode_decode,video_save_path,fps)

    
@torch.no_grad()
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        
    exp_name=  "cat_flower"
    
    image_path = "/home/cby/cogvideo_code/dataset/"+exp_name
    
    decode_path = "/home/cby/cogvideo_code/inversion_output/video_decode/"+exp_name+"_test.mp4"
    output_path = "/home/cby/cogvideo_code/inversion_output/video_recon/"+exp_name+"_test.mp4"
    all_latent_path = "/home/cby/cogvideo_code/inversion_output/all_latent_ddim/"+exp_name+"_test.pt"
    output_edit_path = "/home/cby/cogvideo_code/inversion_output/video_edit/"+exp_name+"_test.mp4"
    
    device="cuda"
    dtype=torch.bfloat16
    
    prompt_dict = {
        #"teaser_car-turn":" A silver jeep driving down a curvy road in the countryside", 
        "teaser_car-turn":"A silver jeep driving along a winding mountain road lined with greenery and tall pine trees. It starts from a distance, with the jeep approaching and coming gradually closer in each frame. As it rounds the bend, the car's details become more visible, including its compact, rugged shape. In the background, tall, rocky mountains add a dramatic backdrop, while the bright daylight enhances the vivid colors of the trees and road. It captures a feeling of exploration and adventure as the jeep progresses along the scenic path.",
        #"swan_swarov":"A graceful black swan glides serenely through a calm pond, its sleek feathers and vibrant red beak creating a striking contrast against the green water and lush foliage. The swan moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the swan's posture and movement, portraying a peaceful moment in nature as the swan explores its pond with calm curiosity.",
        "swan_swarov":"A black swan with a red beak swimming in a river near a wall and bushes.",
        #"swan_swarov":"A black swan with a red beak swimming in a river near a wall and bushes. A black swan with a red beak swimming in a river near a wall and bushes. A black swan with a red beak swimming in a river near a wall and bushes. A black swan with a red beak swimming in a river near a wall and bushes. A black swan with a red beak swimming in a river near a wall and bushes. A black swan with a red beak swimming in a river near a wall and bushes. A black swan with a red beak swimming in a river near a wall and bushes. A black swan with a red beak swimming in a river near a wall and bushes. ",
        "bear_tiger_lion_leopard":"A brown bear walking slowly along a rocky enclosure in a zoo-like setting. The bear's thick fur has a rich, earthy tone that contrasts against the stone walls and rugged rocks surrounding it. Each frame captures a slight change in the bear's posture as it moves forward, suggesting a calm, natural pace. Sunlight filters through, casting soft shadows and highlighting the textures of the bear's fur and the surrounding rocks. The scene has a serene, almost contemplative quality, focusing on the bear's graceful movements and the peaceful atmosphere of its enclosure.",
        
        "surf":"A man is kiteboarding across a vibrant turquoise sea on a sunny day. Wearing a black helmet and colorful shorts, he skillfully maneuvers the board over the water's surface, creating a trail of white spray behind him. His arms are extended as he grips the kite's handle, harnessing the wind to glide forward with ease. The background features distant hills and a faint coastline, adding depth to the scene. The video captures the dynamic motion of kiteboarding, with the man's movements fluid and controlled against the stunning backdrop of open water and clear blue skies.",
        #"surf":"A man is surfing on a white wave",
        
        "squirrel_carrot":"A small, adorable squirrel sits comfortably on a soft yellow cushion, holding a large carrot with both tiny paws. The squirrel nibbles at the carrot, its fluffy tail and pointy ears adding to its endearing appearance. Its little hands clutch the carrot firmly, bringing it closer as it takes delicate bites. The background shows a cozy indoor setting, creating a warm, homely atmosphere. The squirrel's expressions reveal focus and enjoyment as it savors the snack, with its movements gentle and deliberate. The scene is heartwarming and showcases the charm of this tiny creature indulging in its treat.",
        "man_skate":"A construction worker, wearing brown overalls and a blue helmet, is captured in a painting-style scene, performing a daring balancing act on a long wooden plank laid across a set of stairs. The artwork portrays the worker with impressionistic brushstrokes, emphasizing the texture of the fabric and the gleam of the helmet. The setting is a bustling urban environment, rendered in soft, muted tones that create a sense of movement and flow. The worker's arms are outstretched for balance, exuding both skill and concentration. In the background, the steel railing and scattered trees are depicted with delicate detail, adding depth and a hint of realism. This captivating moment combines elements of danger and precision, drawing the viewer into the scene with its artistic interpretation of a precarious situation.",
        "bus_gpu":"A city bus glides smoothly along a tree-lined street, its sleek white and blue exterior reflecting the vibrant urban environment. The scene captures a blend of motion and tranquility as the bus passes by lush green trees and neatly paved sidewalks. Pedestrians can be seen in the background, adding a sense of bustling city life. The gentle sway of the bus and the dappled sunlight filtering through the leaves create a dynamic yet peaceful atmosphere. This moment reflects the harmonious blend of nature and urban transit, showcasing the everyday rhythm of city life.",
        #"child_bike":"A child, wearing a white helmet, rides a small red balance bike along a paved sidewalk surrounded by grass. The child, with a focused expression, grips the handlebars as he carefully balances on the bike without pedals. Each frame captures the child's steady progression, his young face showing determination and curiosity. Sunlight softly illuminates the scene, creating a playful and heartwarming atmosphere.",
        "child_bike":"A young child wearing a teal shirt, black shorts, and a white safety helmet rides a small balance bike along a curved sidewalk in a suburban park. The child grips the handlebars confidently, focusing ahead while learning to maintain balance. The grassy surroundings add a touch of greenery, creating a relaxed outdoor setting. The video captures the slight forward movement, displaying determination and a sense of adventure. The background includes gently sloping lawns and a paved path, emphasizing the everyday charm of a neighborhood stroll while learning to ride.",    
        "fox_wolf_snow":"A fluffy white fox is resting peacefully on a grassy field, surrounded by sparse, delicate stalks of tall grass swaying gently in the breeze. The fox, with its round, soft body and serene expression, appears calm and at ease. As the sequence progresses, the fox slightly shifts its position, occasionally closing its eyes, as if drifting into a comfortable nap. The background is a soft blur of earthy tones, highlighting the fox's thick, pristine fur. The overall atmosphere is tranquil and undisturbed, capturing a peaceful moment in nature.",
        "rabbit":"A black-and-white rabbit with sleek fur sits beside a juicy slice of watermelon on a wooden surface. The rabbit curiously leans toward the vibrant red fruit, sniffing and nibbling delicately at its edge. Its long, upright ears twitch slightly, adding to its lively demeanor. The simple indoor setting features a neutral background, emphasizing the playful interaction between the rabbit and the watermelon. The rabbit's gentle movements and the vivid contrast of the fruit's colors create a charming and refreshing scene. The moment captures the rabbit's natural curiosity and the simple joy of enjoying a sweet summer treat." ,
        "gray_dog":"A fluffy, gray puppy with round, expressive eyes and soft ears resembling bear cubs sits on a cozy beige rug. The puppy tilts its head slightly, radiating curiosity and innocence, as it appears to respond to subtle sounds or movements nearby. The lighting in the room is warm and highlights the puppy's velvety fur, giving it a soft, cuddly look. The background includes faint details of a woven mat and a shadowy figure, adding depth to the setting. The scene captures the endearing charm of the puppy's playful yet calm demeanor, evoking feelings of warmth and delight.",
        "bird_forest":"A vivid scene featuring a scarlet macaw in mid-flight over a tranquil river, its vibrant red, blue, and yellow feathers glistening in the sunlight. The bird gracefully glides against a lush green jungle backdrop, exuding energy and freedom. Its wings spread wide, showcasing intricate patterns as it soars just above the water's surface. The natural setting is serene yet alive with the dynamic motion of the macaw, capturing a perfect harmony of wildlife and landscape. The video evokes a sense of wonder and admiration for the beauty and elegance of tropical ecosystems.",
        "dog_walking":"A cheerful black-and-white Border Collie walks enthusiastically alongside its owner in a quiet suburban neighborhood. The dog, full of energy, looks back at the camera with an alert and happy expression. The man, dressed casually in a navy shirt and jeans, strolls calmly, holding the leash firmly. The scene captures a moment of companionship, framed by a sunny day with a clear blue sky and a house surrounded by lush greenery in the background. The dog's lively demeanor contrasts with the owner's relaxed pace, creating a heartwarming snapshot of their walk.",
        "cat_flower":"A cat, dressed in a black harness with subtle patterns, sits comfortably on the grass under the blossoming cherry tree. It starts by looking to the left. Gradually, the cat turns its head to the right, pausing halfway as if something catches its eye, and then continues the motion, its gaze scanning the surroundings with curiosity. At one point, the cat briefly opens its mouth as if to meow or catch a scent in the air. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The background features a park setting with faint outlines of structures blurred, adding depth to the scene.",
    }
    
    prompt_dict_new = {
        "gray_dog":"A fluffy, gray puppy with soft, bear-like ears sits on a cozy beige rug, its small mouth wide open in an expressive and heartwarming gesture. The puppy's delicate teeth and pink tongue are visible, adding to the charm of this lively moment. Its open mouth suggests a joyful bark, a playful pant, or even a curious response to something exciting nearby. The movement emphasizes its animated and energetic personality, making the scene feel alive. Warm lighting highlights the puppy's velvety fur and the subtle details of its open mouth, drawing the viewer's attention to this captivating and dynamic expression.",
        "bird_forest":"A vivid scene featuring a scarlet macaw in mid-flight over a tranquil river, its vibrant red, blue, and yellow feathers glistening in the sunlight. The bird glides gracefully over the water, frequently turning its head with sharp, fluid motions, scanning the lush jungle surroundings. As it flaps its wings, the parrot turns its head again, shifting its gaze with each movement. The natural setting is serene yet alive, with the macaw's dynamic flight and head turns adding to the sense of energy and freedom. The video captures the harmony of wildlife and landscape, emphasizing the elegance of the tropical ecosystem.",
        "dog_walking":"A cheerful black-and-white Border Collie walks enthusiastically alongside its owner in a quiet suburban neighborhood. The dog, full of energy, repeatedly lowers its head, tilting it slightly as it curiously inspects the ground, its expression alert and focused. With each step, the dog's head lowers again, almost as if tracking something of interest on the path. The man, dressed casually in a navy shirt and jeans, strolls calmly, holding the leash firmly. The scene captures a moment of companionship, framed by a sunny day with a clear blue sky and a house surrounded by lush greenery in the background.",
        "cat_flower":"A cat, dressed in a black harness with subtle patterns, sits comfortably on the grass under the blossoming cherry tree. It starts by looking to the left, its ears twitching as it raises a paw in a fluid motion. The cat then slowly raises its paw higher, as if testing the air, before bringing it down gently. Gradually, the cat turns its head to the right, its raised paw still lingering in the air for a moment as if holding its attention. Pausing halfway, the cat's paw remains suspended before it continues its motion, scanning the surroundings with curiosity. At one point, the cat briefly opens its mouth as if to meow or catch a scent in the air, its paw now resting gracefully on the ground.",
    }
        
    prompt = prompt_dict[exp_name]
    prompt_new = prompt_dict_new[exp_name]
    num_inference_steps=50
    guidance_scale=6
    num_frames=9
    fps=4

    
    # #load the cogvideo-2b
    # pipe = CogVideoXPipeline.from_pretrained("/mnt/nfs/CogVideoX-2b",torch_dtype=dtype).to(device)
    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    # pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    
    #load the cogvideo-5b
    pipe = CogVideoXPipeline.from_pretrained("/mnt/nfs/CogVideoX-5b", torch_dtype=dtype,device_map="balanced")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    
    # #load the cogvideo-5b
    # pipe = CogVideoXPipeline.from_pretrained("/mnt/nfs/CogVideoX-5b",torch_dtype=dtype).to(device)
    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()
    # pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
    

    latent=transfer_image_to_latent(pipe,image_path,decode_path,dtype = dtype)    
    
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds=pipe.encode_prompt(prompt,device=device)
    
    #ddim inversion
    # model_patch(pipe)
    # global output_cross 
    
    all_latent=ddim_loop(pipe,latent,num_inference_steps,prompt_embeds,negative_prompt_embeds)
    torch.save(all_latent,all_latent_path)
    
    # lb = LocalBlend([prompt],("man",),pipe.tokenizer)
    # mask = lb(output_cross).cpu().squeeze()
    # for i in range(mask.shape[0]):
    #     # mask = mask.gt(0.3)
    #     plt.imshow(mask[i])
    #     plt.savefig(f"ddim_{i}.jpg")

    
    #ddim recon
    generate_video_with_latent(pipe,prompt,output_path,num_inference_steps,all_latent[-1],guidance_scale,num_frames,fps)
    #prompt = "A graceful white duck glides serenely through a calm pond, its soft feathers and bright orange beak creating a striking contrast against the green water and lush foliage. The duck moves with quiet elegance, occasionally dipping its head and neck into the water, leaving gentle ripples in its wake. The surrounding vegetation, rich and green, reflects in the water, adding to the tranquil atmosphere of the scene. Each frame captures subtle shifts in the duck's posture and movement, portraying a peaceful moment in nature as the duck explores its pond with calm curiosity."
    #prompt = "A panda walking slowly along a rocky enclosure in a zoo-like setting. The panda’s black-and-white fur stands out against the stone walls and rugged rocks around it, creating a striking contrast. Each frame shows a subtle shift in the panda’s posture as it moves forward at a calm, deliberate pace. Sunlight filters through, casting gentle shadows and illuminating the textures of the panda's fur and the surrounding rocks. The scene feels serene and tranquil, emphasizing the panda's gentle movements and the peaceful atmosphere of its naturalistic enclosure."
    #prompt = "Spider-Man is kiteboarding across a vibrant turquoise sea on a sunny day. Dressed in his iconic red and blue suit, he skillfully maneuvers the board over the water’s surface, leaving a trail of white spray behind him. His arms are extended as he grips the kite's handle, using his super agility to harness the wind and glide forward with ease. The background features distant hills and a faint coastline, adding depth to the scene. The video captures the dynamic, action-packed motion of Spider-Man kiteboarding, his movements fluid and controlled against the stunning backdrop of open water and clear blue skies."
    #prompt = "A small, futuristic robot mouse sits on a soft yellow cushion, holding a large carrot with its metallic paws. The robot’s design is sleek and cute, with tiny gears and LED eyes that give it an expressive appearance. Its mechanical hands grip the carrot, bringing it closer as it performs a gentle, nibbling motion, mimicking a real mouse’s eating behavior. The background shows a cozy indoor setting, enhancing the contrast between the soft surroundings and the robot's metallic frame. The robot mouse’s movements are precise and smooth, creating a charming scene as it “enjoys” its snack in an unexpectedly endearing way."
    #prompt = "Wonder Woman, wearing her iconic costume, attempts a daring balancing act on a long wooden plank laid across a set of stairs. The scene unfolds in a bustling urban environment, with her carefully maneuvering to maintain stability on the narrow beam. With arms stretched out for balance, she shifts weight from side to side, displaying both skill and concentration. The background reveals a steel railing and a few scattered trees, adding depth to the scene. This captivating moment combines elements of danger and precision, capturing the viewer's attention as Wonder Woman deftly navigates the precarious situation."
    #prompt = "In a vibrant cityscape scene, a city bus painted in striking white and blue hues moves gracefully down a bustling street. The setting is depicted in a watercolor style, with soft, flowing brushstrokes capturing the motion and energy of urban life. Trees lining the street are rendered in lush greens, their leaves blending into the background. The bus glides past, its form slightly blurred to convey speed. Pedestrians are suggested with light dabs of color, adding to the sense of movement and activity. The overall atmosphere is dynamic and lively, highlighting the seamless integration of public transport within the city's vibrant pulse."
    #prompt = "A white duck with a red beak swimming in a river near a wall and bushes."
    #prompt = "A panda, with black-and-white fur, rides a small red balance bike along a paved sidewalk surrounded by grass. The panda, with a focused expression, grips the handlebars as it carefully balances on the bike without pedals. Each frame captures the panda's steady progression, its fluffy face showing determination and curiosity. Sunlight softly illuminates the scene, creating a playful and heartwarming atmosphere."
    #prompt = "A panda rides a small balance bike along a curved sidewalk in a suburban park. The panda grips the handlebars with its fluffy paws, focusing ahead while trying to maintain balance on the bike. Its furry face shows determination and a hint of playful curiosity. The grassy surroundings add a natural touch, creating a charming outdoor setting. Each frame captures the panda's slight forward movement, highlighting its adventurous spirit. The background includes gently sloping lawns and a paved path, enhancing the playful and whimsical nature."
    #prompt = "Iron Man is surfing on a white wave"
    #prompt = "Iron Man is kiteboarding across a vibrant turquoise sea on a sunny day. Wearing his iconic red and gold suit and a black helmet, he skillfully maneuvers the board over the water's surface, creating a trail of white spray behind him. His arms are extended as he grips the kite's handle, harnessing the wind to glide forward with ease. The background features distant hills and a faint coastline, adding depth to the scene. The video captures the dynamic motion of kiteboarding, with Iron Man's movements fluid and controlled against the stunning backdrop of open water and clear blue skies."
    #prompt = "A majestic tiger, with a thick, warm coat of orange fur marked by bold black stripes, is resting peacefully on a grassy field. Sparse, tall stalks of grass sway gently in the breeze around it, framing the tiger as it lies with a calm, relaxed expression. Occasionally, the tiger shifts its position, closing its eyes as if settling into a comfortable nap. The background is a soft blur of earthy tones, enhancing the vivid colors and strength of the tiger's form. The atmosphere is serene, capturing a rare, peaceful moment in the life of this powerful predator."
    #prompt = "A sleek orange tabby cat with glossy fur sits beside a juicy slice of watermelon on a wooden surface. The cat curiously leans toward the vibrant red fruit, sniffing and gently pawing at its edge. Its sharp, alert ears flick slightly, adding to its lively demeanor. The simple indoor setting features a neutral background, emphasizing the playful interaction between the cat and the watermelon. The cat's graceful movements and the vivid contrast of the fruit's colors create a charming and refreshing scene. The moment captures the cat's natural curiosity and the simple joy of exploring a sweet summer treat."
    generate_video_with_latent(pipe,prompt_new,output_edit_path,num_inference_steps,all_latent[-1],guidance_scale,num_frames,fps)
    pdb.set_trace()
    

if __name__=='__main__':
    main()