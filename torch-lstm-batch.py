#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
from glob import glob
import pandas as pd
import linecache


# In[ ]:


datasets = pd.read_csv("/content/drive/MyDrive/catch-up/torch-lstm/datasets.csv")


# In[ ]:


datasets.head()


# In[ ]:


categories = set(datasets["category"])


# In[ ]:


categories


# In[ ]:


len(datasets)


# In[ ]:


import torch
import torch.nn as nn
embeds = nn.Embedding(10, 6) # (Embedding(単語の合計数, ベクトル次元数))
w1 = torch.tensor([2])
print(embeds(w1))
w2 = torch.tensor([2,4,9])
print(embeds(w2))


# In[ ]:


get_ipython().system('apt install aptitude swig')
get_ipython().system('aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y')
# 以下で報告があるようにmecab-python3のバージョンを0.996.5にしないとtokezerで落ちる
# https://stackoverflow.com/questions/62860717/huggingface-for-japanese-tokenizer
get_ipython().system('pip install mecab-python3==0.996.5')
get_ipython().system('pip install unidic-lite # これないとMeCab実行時にエラーで落ちる')


# In[ ]:


import MeCab
import re
import torch

tagger = MeCab.Tagger("-Owakati")

def make_wakati(sentence):
    # MeCabで分かち書き
    sentence = tagger.parse(sentence)
    # 半角全角英数字除去
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # 記号もろもろ除去
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    # スペースで区切って形態素の配列へ
    wakati = sentence.split(" ")
    # 空の要素は削除
    wakati = list(filter(("").__ne__, wakati))
    return wakati

# テスト
test = "【人工知能】は「人間」の仕事を奪った"
print(make_wakati(test))
# ['人工', '知能', 'は', '人間', 'の', '仕事', 'を', '奪っ', 'た']


# In[ ]:


# 単語ID辞書を作成する
word2index = {}
word2index.update({"<pad>":0})
for title in datasets["title"]:
    wakati = make_wakati(title)
    for word in wakati:
        if word in word2index: continue
        word2index[word] = len(word2index)
print("vocab size : ", len(word2index))
print(word2index)


# In[ ]:


from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle


# In[ ]:


cat2index = {}
for cat in categories:
    if cat in cat2index: continue
    cat2index[cat] = len(cat2index)
print(cat2index)


# In[ ]:


def sentence2index(sentence):
    wakati = make_wakati(sentence)
    return [word2index[w] for w in wakati]

def category2index(cat):
    return [cat2index[cat]]


# In[ ]:


index_datasets_title_tmp = []
index_datasets_category = []


# ### 系列の長さの最大値を取得。この長さに他の系列の長さをあわせる

# In[ ]:


max_len = 0


# In[ ]:


for title, category in zip(datasets["title"], datasets["category"]):
    # print(title)
    # print(category)
    index_title = sentence2index(title)
    index_category = category2index(category)
    # print(index_title)
    # print(index_category)
    # 文字列を数値にindexに変換することは大切なのか...
    index_datasets_title_tmp.append(index_title)
    index_datasets_category.append(index_category)
    if max_len < len(index_title):
        max_len = len(index_title)


# ### 系列の長さを揃えるために短い系列にパディングを追加
# ### 後ろパディングだと正しく学習できなかったので、前パディング

# In[ ]:


index_datasets_title = []
for title in index_datasets_title_tmp:
  for i in range(max_len - len(title)):
    title.insert(0, 0) # 前パディング
#     title.append(0)　# 後ろパディング
  index_datasets_title.append(title)
print(index_datasets_title)


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(index_datasets_title, index_datasets_category, train_size=0.7)


# In[ ]:


# データをバッチでまとめるための関数
def train2batch(title, category, batch_size=100):
  title_batch = []
  category_batch = []
  title_shuffle, category_shuffle = shuffle(title, category)
  for i in range(0, len(title), batch_size):
    title_batch.append(title_shuffle[i:i+batch_size])
    category_batch.append(category_shuffle[i:i+batch_size])
  return title_batch, category_batch


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[ ]:


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # <pad>の単語IDが0なので、padding_idx=0としている
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # batch_first=Trueが大事！
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        #embeds.size() = (batch_size × len(sentence) × embedding_dim)
        _, lstm_out = self.lstm(embeds)
        # lstm_out[0].size() = (1 × batch_size × hidden_dim)
        tag_space = self.hidden2tag(lstm_out[0])
        # tag_space.size() = (1 × batch_size × tagset_size)

        # (batch_size × tagset_size)にするためにsqueeze()する
        tag_scores = self.softmax(tag_space.squeeze())
        # tag_scores.size() = (batch_size × tagset_size)

        return tag_scores


# In[ ]:


EMBEDDING_DIM = 200
HIDDEN_DIM = 128
VOCAB_SIZE = len(word2index)
TAG_SIZE = len(categories)
# to(device)でモデルがGPU対応する
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE).to(device)
loss_function = nn.NLLLoss()
# SGDからAdamに変更。特に意味はなし
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


losses = []
for epoch in range(100):
    all_loss = 0
    title_batch, category_batch = train2batch(train_x, train_y)
    for i in range(len(title_batch)):
        batch_loss = 0

        model.zero_grad()

        # 順伝搬させるtensorはGPUで処理させるためdevice=にGPUをセット
        title_tensor = torch.tensor(title_batch[i], device=device)
        # category_tensor.size() = (batch_size × 1)なので、squeeze()
        category_tensor = torch.tensor(category_batch[i], device=device).squeeze()

        out = model(title_tensor)

        batch_loss = loss_function(out, category_tensor)
        batch_loss.backward()
        optimizer.step()

        all_loss += batch_loss.item()
    print("epoch", epoch, "\t" , "loss", all_loss)
    if all_loss < 0.1: break
print("done.")


# In[ ]:


test_num = len(test_x)
a = 0
with torch.no_grad():
    title_batch, category_batch = train2batch(test_x, test_y)

    for i in range(len(title_batch)):
        title_tensor = torch.tensor(title_batch[i], device=device)
        category_tensor = torch.tensor(category_batch[i], device=device)

        out = model(title_tensor)
        _, predicts = torch.max(out, 1)
        for j, ans in enumerate(category_tensor):
            if predicts[j].item() == ans.item():
                a += 1
print("predict : ", a / test_num)
# めちゃ精度高くない？？？
# predict :  0.6967916854948034


# In[ ]:




