#!/usr/bin/env python
# coding: utf-8

# # Lesson 3 - Recommender Systems

# ### Import the Needed Packages

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm, trange
from DLAIUtils import Utils

import pandas as pd
import time
import os


# In[ ]:


utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()
OPENAI_API_KEY = utils.get_openai_api_key()


# ### Load the Dataset
# 
# **Note:** To access the dataset outside of this course, just copy the following two lines of code and run it (remember to uncomment them first before executing):
# 
# !wget -q --show-progress -O all-the-news-3.zip "https://www.dropbox.com/scl/fi/wruzj2bwyg743d0jzd7ku/all-the-news-3.zip?rlkey=rgwtwpeznbdadpv3f01sznwxa&dl=1"
# 
# !unzip all-the-news-3.zip

# In[ ]:


with open('./data/all-the-news-3.csv', 'r') as f:
    header = f.readline()
    print(header)


# In[ ]:


df = pd.read_csv('./data/all-the-news-3.csv', nrows=99)
df.head()


# ### Setup Pinecone

# In[ ]:


openai_client = OpenAI(api_key=OPENAI_API_KEY)
util = Utils()
INDEX_NAME = utils.create_dlai_index_name('dl-ai')
pinecone = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
  pinecone.delete_index(INDEX_NAME)

pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine',
  spec=ServerlessSpec(cloud='aws', region='us-west-2'))

index = pinecone.Index(INDEX_NAME)


# ### 1.  Create Embeddings of the News Titles

# In[ ]:


def get_embeddings(articles, model="text-embedding-ada-002"):
   return openai_client.embeddings.create(input = articles, model=model)


# In[ ]:


CHUNK_SIZE=400
TOTAL_ROWS=10000
progress_bar = tqdm(total=TOTAL_ROWS)
chunks = pd.read_csv('./data/all-the-news-3.csv', chunksize=CHUNK_SIZE, 
                     nrows=TOTAL_ROWS)
chunk_num = 0
for chunk in chunks:
    titles = chunk['title'].tolist()
    embeddings = get_embeddings(titles)
    prepped = [{'id':str(chunk_num*CHUNK_SIZE+i), 'values':embeddings.data[i].embedding,
                'metadata':{'title':titles[i]},} for i in range(0,len(titles))]
    chunk_num = chunk_num + 1
    if len(prepped) >= 200:
      index.upsert(prepped)
      prepped = []
    progress_bar.update(len(chunk))


# In[ ]:


index.describe_index_stats()


# ### Build the Recommender System

# In[ ]:


def get_recommendations(pinecone_index, search_term, top_k=10):
  embed = get_embeddings([search_term]).data[0].embedding
  res = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True)
  return res


# In[ ]:


reco = get_recommendations(index, 'obama')
for r in reco.matches:
    print(f'{r.score} : {r.metadata["title"]}')


# ### 2.  Create Embeddings of All News Content

# In[ ]:


if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
  pinecone.delete_index(name=INDEX_NAME)

pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine',
  spec=ServerlessSpec(cloud='aws', region='us-west-2'))
articles_index = pinecone.Index(INDEX_NAME)


# In[ ]:


def embed(embeddings, title, prepped, embed_num):
  for embedding in embeddings.data:
    prepped.append({'id':str(embed_num), 'values':embedding.embedding, 'metadata':{'title':title}})
    embed_num += 1
    if len(prepped) >= 100:
        articles_index.upsert(prepped)
        prepped.clear()
  return embed_num


# <p style="background-color:#fff1d7; padding:15px; "> <b>(Note: <code>news_data_rows_num = 100</code>):</b> In this lab, we've initially set <code>news_data_rows_num</code> to 100 for speedier results, allowing you to observe the outcomes faster. Once you've done an initial run, consider increasing this value to 200, 400, 700, and 1000. You'll likely notice better and more relevant results.</p>

# In[ ]:


news_data_rows_num = 100

embed_num = 0 #keep track of embedding number for 'id'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, 
    chunk_overlap=20) # how to chunk each article
prepped = []
df = pd.read_csv('./data/all-the-news-3.csv', nrows=news_data_rows_num)
articles_list = df['article'].tolist()
titles_list = df['title'].tolist()

for i in range(0, len(articles_list)):
    print(".",end="")
    art = articles_list[i]
    title = titles_list[i]
    if art is not None and isinstance(art, str):
      texts = text_splitter.split_text(art)
      embeddings = get_embeddings(texts)
      embed_num = embed(embeddings, title, prepped, embed_num)


# In[ ]:


articles_index.describe_index_stats()


# ### Build the Recommender System

# In[ ]:


reco = get_recommendations(articles_index, 'obama', top_k=100)
seen = {}
for r in reco.matches:
    title = r.metadata['title']
    if title not in seen:
        print(f'{r.score} : {title}')
        seen[title] = '.'


# In[ ]:




