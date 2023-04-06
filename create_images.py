#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycocotools.coco import COCO
import pandas as pd

# In[4]:
texts = pd.read_csv('extended_prompts.csv')
texts = texts['extended'].values[10000:20000]
print(texts)

# In[5]:


from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(torch_dtype=torch.float16,
                                               revision="fp16",
                                               pretrained_model_name_or_path='../../stablediffusion/stable-diffusion-2')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


# In[6]:


pipe.enable_xformers_memory_efficient_attention()


# In[ ]:


import hashlib
from tqdm import tqdm

batch_size = 8

img_paths = []
prompts = []

for i in tqdm(range(len(texts) // batch_size)):
    text_batch = texts[i:i+batch_size].tolist()
    with torch.autocast("cuda"): 
        output = pipe(text_batch, num_images_per_prompt=1)
        images = output.images
    
    for i in range(len(text_batch)):
        text_hash = hashlib.sha256(f"{text_batch[i]}_{i}".encode('utf-8')).hexdigest()
        output_path = f"coco_gen_ex_2/{text_hash}.png"
        images[i].save(f"coco_gen_ex_2/{text_hash}.png")
        print(text_hash)
        img_paths.append(output_path)
        prompts.append(text_batch[i])


# In[ ]:


d = {'path': prompts, 'prompt': img_paths}
pd.DataFrame(data=d).to_csv('files_ex_2.csv', index=False)


