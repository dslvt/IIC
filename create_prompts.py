# %%
from pycocotools.coco import COCO
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import torch


annFile = './annotations/captions_train2017.json'
coco_caps=COCO(annFile)
vals  = coco_caps.anns.values()
texts = []
for v in list(vals):
    texts.append(v['caption'])


# %%

texts_df = {"text": texts}
texts_df = Dataset.from_dict(texts_df)
texts_df = texts_df.map(lambda examples: examples, batched=True)


# %%
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='daspartho/prompt-extend', device=0)
set_seed(42)

# %%
from transformers.pipelines.pt_utils import KeyDataset

v = []

for out in tqdm(generator(KeyDataset(texts_df, "text"), 
                          batch_size=160, 
                          max_length=50, 
                          num_return_sequences=5, 
                          pad_token_id=generator.tokenizer.eos_token_id)):
    v.append(out)

# %%
generated_texts = []
original_texts = []

for i, gen_batch in enumerate(v):
    for gen_text in gen_batch:
        generated_texts.append(gen_text['generated_text'])
        original_texts.append(texts[i])

# %%
pd.DataFrame(data={'text': original_texts, "extended": generated_texts}).to_csv('extended_prompts.csv', index=False)

# %%



