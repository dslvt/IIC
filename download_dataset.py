#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycocotools.coco import COCO


# In[2]:


annFile = './annotations/captions_train2017.json'
coco_caps=COCO(annFile)


# In[3]:


vals  = coco_caps.anns.values()
texts = []
for v in list(vals):
    texts.append(v['caption'])


# In[4]:


texts


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

img_paths = []
prompts = []

for text in tqdm(texts[:3103]):
    print(text)
    with torch.autocast("cuda"): 
        output = pipe(text, num_images_per_prompt=1, height=512, width=512, num_inference_steps=20)
        images = output.images
    
    for i in range(len(images)):
        text_hash = hashlib.sha256(f"{text}_{i}".encode('utf-8')).hexdigest()
        output_path = f"coco_gen_512/{text_hash}.png"
        images[i].save(f"coco_gen_512/{text_hash}.png")
        img_paths.append(output_path)
        prompts.append(text)


# In[ ]:


import pandas as pd


# In[ ]:


d = {'path': prompts, 'prompt': img_paths}
pd.DataFrame(data=d).to_csv('files.csv', index=False)


