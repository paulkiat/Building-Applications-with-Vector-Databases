#!/usr/bin/env python
# coding: utf-8

# # Lesson 2 - Retrieval Augmented Generation (RAG)

# ### Import  the Needed Packages

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from datasets import load_dataset
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from DLAIUtils import Utils

import ast
import os
import pandas as pd


# In[ ]:


# get api key
utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()


# ### Setup Pinecone

# In[ ]:


pinecone = Pinecone(api_key=PINECONE_API_KEY)

utils = Utils()
INDEX_NAME = utils.create_dlai_index_name('dl-ai')
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
  pinecone.delete_index(INDEX_NAME)

pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine',
  spec=ServerlessSpec(cloud='aws', region='us-west-2'))

index = pinecone.Index(INDEX_NAME)


# ### Load the Dataset
# 
# **Note:** To access the dataset outside of this course, just copy the following two lines of code and run it (remember to uncomment them first before executing):
# 
# #!wget -q -O lesson2-wiki.csv.zip "https://www.dropbox.com/scl/fi/yxzmsrv2sgl249zcspeqb/lesson2-wiki.csv.zip?rlkey=paehnoxjl3s5x53d1bedt4pmc&dl=0"
# 
# #!unzip lesson2-wiki.csv.zip

# <p style="background-color:#fff1d7; padding:15px; "> <b>(Note: <code>max_articles_num = 500</code>):</b> To achieve a more comprehensive context for the Language Learning Model, a larger number of articles is generally more beneficial. In this lab, we've initially set <code>max_articles_num</code> to 500 for speedier results, allowing you to observe the outcomes faster. Once you've done an initial run, consider increasing this value to 750 or 1,000. You'll likely notice that the context provided to the LLM becomes richer and better. You can experiment by gradually raising this variable for different queries to observe the improvements in the LLM's contextual understanding.</p>

# In[ ]:


max_articles_num = 500
df = pd.read_csv('./data/wiki.csv', nrows=max_articles_num)
df.head()


# ### Prepare the Embeddings and Upsert to Pinecone

# In[ ]:


prepped = []

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    meta = ast.literal_eval(row['metadata'])
    prepped.append({'id':row['id'], 
                    'values':ast.literal_eval(row['values']), 
                    'metadata':meta})
    if len(prepped) >= 250:
        index.upsert(prepped)
        prepped = []


# In[ ]:


index.describe_index_stats()


# ### Connect to OpenAI

# In[ ]:


OPENAI_API_KEY = utils.get_openai_api_key()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(articles, model="text-embedding-ada-002"):
   return openai_client.embeddings.create(input = articles, model=model)


# ### Run Your Query

# In[ ]:


query = "what is the berlin wall?"

embed = get_embeddings([query])
res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)
text = [r['metadata']['text'] for r in res['matches']]
print('\n'.join(text))


# ### Build the Prompt

# In[ ]:


query = "write an article titled: what is the berlin wall?"
embed = get_embeddings([query])
res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)

contexts = [
    x['metadata']['text'] for x in res['matches']
]

prompt_start = (
    "Answer the question based on the context below.\n\n"+
    "Context:\n"
)

prompt_end = (
    f"\n\nQuestion: {query}\nAnswer:"
)

prompt = (
    prompt_start + "\n\n---\n\n".join(contexts) + 
    prompt_end
)

print(prompt)


# ### Get the Summary 

# In[ ]:


res = openai_client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    temperature=0,
    max_tokens=636,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
)
print('-' * 80)
print(res.choices[0].text)


# In[ ]:




