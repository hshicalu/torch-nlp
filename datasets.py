#!/usr/bin/env python
# coding: utf-8

# ### データセットをdfに変換する
# Colabでは実装に30分ほどかかるので、csvに保存しておく

# In[1]:


import os
from glob import glob
import pandas as pd
import linecache


# In[5]:


# カテゴリを配列で取得
categories = [name for name in os.listdir("text/") if os.path.isdir("text/" + name)]
print(categories)


# In[9]:


datasets = pd.DataFrame(columns=["title", "category"])
for cat in categories:
    path = "text/" + cat + "/*.txt"
    print(path)
    files = glob(path)
    for text_name in files:
        title = linecache.getline(text_name, 3)
        s = pd.Series([title, cat], index=datasets.columns)
        datasets = datasets.append(s, ignore_index=True)

# データフレームシャッフル
datasets = datasets.sample(frac=1).reset_index(drop=True)
datasets.head()


# In[12]:


len(datasets)


# In[13]:


datasets.to_csv("datasets.csv")


# In[ ]:




