#!/usr/bin/env python
# coding: utf-8

# # Lesson 1 - Semantic Search

# Welcome to Lesson 1. 
# 
# To access the `requirement.txt` file, go to `File` and click on `Open`.
#  
# I hope you enjoy this course!

# ### Import the Needed Packages

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from DLAIUtils import Utils
import DLAIUtils

import os
import time
import torch


# In[ ]:


from tqdm.auto import tqdm


# ### Load the Dataset

# In[ ]:


dataset = load_dataset('quora', split='train[240000:290000]')


# In[ ]:


dataset[:5]


# In[ ]:


questions = []
for record in dataset['questions']:
    questions.extend(record['text'])
question = list(set(questions))
print('\n'.join(questions[:10]))
print('-' * 50)
print(f'Number of questions: {len(questions)}')


# ### Check cuda and Setup the model
# 
# **Note**: "Checking cuda" refers to checking if you have access to GPUs (faster compute). In this course, we are using CPUs. So, you might notice some code cells taking a little longer to run.
# 
# We are using *all-MiniLM-L6-v2* sentence-transformers model that maps sentences to a 384 dimensional dense vector space.

# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print('Sorry no cuda.')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


# In[ ]:


query = 'which city is the most populated in the world?'
xq = model.encode(query)
xq.shape


# ### Setup Pinecone

# In[ ]:


utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()


# In[ ]:


pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = utils.create_dlai_index_name('dl-ai')

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)
print(INDEX_NAME)
pinecone.create_index(name=INDEX_NAME, 
    dimension=model.get_sentence_embedding_dimension(), 
    metric='cosine',
    spec=ServerlessSpec(cloud='aws', region='us-west-2'))

index = pinecone.Index(INDEX_NAME)
print(index)


# ### Create Embeddings and Upsert to Pinecone

# In[ ]:


batch_size=200
vector_limit=10000

questions = question[:vector_limit]

import json

for i in tqdm(range(0, len(questions), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(questions))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadatas = [{'text': text} for text in questions[i:i_end]]
    # create embeddings
    xc = model.encode(questions[i:i_end])
    # create records list for upsert
    records = zip(ids, xc, metadatas)
    # upsert to Pinecone
    index.upsert(vectors=records)


# In[ ]:


index.describe_index_stats()


# ### Run Your Query

# In[ ]:


# small helper function so we can repeat queries later
def run_query(query):
  embedding = model.encode(query).tolist()
  results = index.query(top_k=10, vector=embedding, include_metadata=True, include_values=False)
  for result in results['matches']:
    print(f"{round(result['score'], 2)}: {result['metadata']['text']}")


# In[ ]:


run_query('which city has the highest population in the world?')


# In[ ]:


query = 'how do i make chocolate cake?'
run_query(query)

