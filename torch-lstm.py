#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
from glob import glob
import pandas as pd
import linecache


# In[ ]:


# # カテゴリを配列で取得
# categories = [name for name in os.listdir("/content/drive/MyDrive/catch-up/torch-lstm/text/") if os.path.isdir("/content/drive/MyDrive/catch-up/torch-lstm/text/" + name)]
# print(categories)


# In[ ]:


# datasets = pd.DataFrame(columns=["title", "category"])
# for cat in categories:
#     path = "/content/drive/MyDrive/catch-up/torch-lstm/text/" + cat + "/*.txt"
#     print(path)
#     files = glob(path)
#     for text_name in files:
#         title = linecache.getline(text_name, 3)
#         s = pd.Series([title, cat], index=datasets.columns)
#         datasets = datasets.append(s, ignore_index=True)

# # データフレームシャッフル
# datasets = datasets.sample(frac=1).reset_index(drop=True)
# datasets.head()


# In[ ]:


datasets = pd.read_csv("/content/drive/MyDrive/catch-up/torch-lstm/datasets.csv")
categories = set(datasets["category"])


# In[ ]:


datasets.head()


# In[ ]:


categories


# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


embeds = nn.Embedding(10, 6) # (Embedding(単語の合計数, ベクトル次元数))


# In[ ]:


w1 = torch.tensor([2])
print(embeds(w1))


# In[ ]:


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

# 単語ID辞書を作成する
word2index = {}
for title in datasets["title"]:
    wakati = make_wakati(title)
    for word in wakati:
        if word in word2index: continue
        word2index[word] = len(word2index)
print("vocab size : ", len(word2index))


# In[ ]:


def sentence2index(sentence):
    wakati = make_wakati(sentence)
    print(wakati)
    return torch.tensor([word2index[w] for w in wakati], dtype=torch.long)

# テスト
test = "例のあのメニューも！ニコニコ超会議のフードコートメニュー14種類紹介（前半）"
print(sentence2index(test))


# In[ ]:


VOCAB_SIZE = len(word2index)
EMBEDDING_DIM = 10
test = "ユージの前に立ちはだかったJOY「僕はAKBの高橋みなみを守る」"
inputs = sentence2index(test)
embeds = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
sentence_matrix = embeds(inputs)
print(sentence_matrix.size())
print(sentence_matrix)


# In[ ]:


sentence_matrix.view(len(sentence_matrix), 1, -1).size()


# In[ ]:


VOCAB_SIZE = len(word2index)
EMBEDDING_DIM = 10
HIDDEN_DIM = 128
embeds = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)
s1 = "震災をうけて感じた、大切だと思ったこと"
print(make_wakati(s1))
#['震災', 'を', 'うけ', 'て', '感じ', 'た', '大切', 'だ', 'と', '思っ', 'た', 'こと']


# In[ ]:


inputs1 = sentence2index(s1)
emb1 = embeds(inputs1)
lstm_inputs1 = emb1.view(len(inputs1), 1, -1)
out1, out2 = lstm(lstm_inputs1)
print(out1)
print(out2)


# In[ ]:


# class LSTMClassifier(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
#         # 親クラスのコンストラクタ。決まり文句
#         super(LSTMClassifier, self).__init__()
#         # 隠れ層の次元数。これは好きな値に設定しても行列計算の過程で出力には出てこないので。
#         self.hidden_dim = hidden_dim
#         # インプットの単語をベクトル化するために使う
#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
#         # LSTMの隠れ層。これ１つでOK。超便利。
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)
#         # LSTMの出力を受け取って全結合してsoftmaxに食わせるための１層のネットワーク
#         self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
#         # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
#         self.softmax = nn.LogSoftmax(dim=1)
    
#     def forward(self, sentence):
#         # 文章内の単語をベクトル化して出力する
#         embeds = self.word_embeddings(sentence)
#         # lstmに入れるように成形する
#         # 第二戻り値のみの書き方こんなんなんや...
#         _, letm_out = self.lstm(embeds.view(len(sentence), 1, -1))
#         tag_space = self.hidden2tag(lstm_out[0].view(-1, self.hidden_dim))
#         tag_scores = self.softmax(tag_space)
#         return tag_scores


# In[ ]:


# nn.Moduleを継承して新しいクラスを作る。決まり文句
class LSTMClassifier(nn.Module):
    # モデルで使う各ネットワークをコンストラクタで定義
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        # 親クラスのコンストラクタ。決まり文句
        super(LSTMClassifier, self).__init__()
        # 隠れ層の次元数。これは好きな値に設定しても行列計算の過程で出力には出てこないので。
        self.hidden_dim = hidden_dim
        # インプットの単語をベクトル化するために使う
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTMの隠れ層。これ１つでOK。超便利。
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # LSTMの出力を受け取って全結合してsoftmaxに食わせるための１層のネットワーク
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # softmaxのLog版。dim=0で列、dim=1で行方向を確率変換。
        self.softmax = nn.LogSoftmax(dim=1)

    # 順伝播処理はforward関数に記載
    def forward(self, sentence):
        # 文章内の各単語をベクトル化して出力。2次元のテンソル
        embeds = self.word_embeddings(sentence)
        # 2次元テンソルをLSTMに食わせられる様にviewで３次元テンソルにした上でLSTMへ流す。
        # 上記で説明した様にmany to oneのタスクを解きたいので、第二戻り値だけ使う。
        _, lstm_out = self.lstm(embeds.view(len(sentence), 1, -1))
        # lstm_out[0]は３次元テンソルになってしまっているので2次元に調整して全結合。
        tag_space = self.hidden2tag(lstm_out[0].view(-1, self.hidden_dim))
        # softmaxに食わせて、確率として表現
        tag_scores = self.softmax(tag_space)
        return tag_scores


# In[ ]:


category2index = {}
for cat in categories:
    if cat in category2index: continue
    category2index[cat] = len(category2index)

print(category2index)

def category2tensor(cat):
    return torch.tensor([category2index[cat]], dtype = torch.long)

print(category2tensor("it-life-hack"))


# In[ ]:


from sklearn.model_selection import train_test_split
import torch.optim as optim
train, test = train_test_split(datasets, train_size = .7)


# In[ ]:


# 単語のベクトル次元数
EMBEDDING_DIM = 10
# 隠れ層の次元数
HIDDEN_DIM = 128
# データ全体の単語数
VOCAB_SIZE = len(word2index)
# 分類先のカテゴリの数
TAG_SIZE = len(categories)


# In[ ]:


# モデル宣言
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
# 損失関数はNLLLoss()を使う。LogSoftmaxを使う時はこれを使うらしい。
loss_function = nn.NLLLoss()
# 最適化の手法はSGDで。lossの減りに時間かかるけど、一旦はこれを使う。
optimizer = optim.SGD(model.parameters(), lr=0.01)


# ## GPUに切り替えてから学習を実行する

# In[ ]:


losses = []

for epoch in range(100):
    all_loss = 0
    for title, cat in zip(train["title"], train["category"]):
        # 勾配の情報をリセット
        model.zero_grad()
        # 文章を単語IDの系列に変換
        inputs = sentence2index(title)
        # 順伝搬の結果をget
        out = model(inputs)
        # 正解カテゴリをテンソル化
        answer = category2tensor(cat)
        # lossの計算
        loss = loss_function(out, answer)
        # 勾配をリセット
        loss.backward()
        # 逆伝搬でパラメータを更新する
        optimizer.step()
        # lossを集計
        all_loss += loss.item()
    losses.append(all_loss)
    print("epoch", epoch, "\t", "loss", all_loss)
print("done.")


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


plt.plot(losses)
plt.show()


# In[ ]:


# テストデータの母数計算
test_num = len(test)
# 正解の件数
a = 0
# 勾配自動計算OFF
with torch.no_grad():
    for title, category in zip(test["title"], test["category"]):
        # テストデータの予測
        inputs = sentence2index(title)
        out = model(inputs)

        # outの一番大きい要素を予測結果をする
        _, predict = torch.max(out, 1)

        answer = category2tensor(category)
        if predict == answer:
            a += 1
print("predict : ", a / test_num)
# predict :  0.6118391323994578


# In[ ]:


traindata_num = len(train)
a = 0
with torch.no_grad():
    for title, category in zip(train["title"], train["category"]):
        inputs = sentence2index(title)
        out = model(inputs)
        _, predict = torch.max(out, 1)
        answer = category2tensor(category)
        if predict == answer:
            a += 1
print("predict : ", a / traindata_num)
# predict :  0.9984505132674801


# ## 過学習しているね！！！

# In[ ]:


import collections
index2category = {}
for cat, idx in category2index.items():
    index2category[idx] = cat


# In[ ]:


predict_df = pd.DataFrame(columns=["answer", "predict", "exact"])


# In[ ]:


with torch.no_grad():
    for title, category in zip(test["title"], test["category"]):
        out = model(sentence2index(title))
        _, predict = torch.max(out, 1)
        answer = category2tensor(category)
        exact = "O" if predict.item() == answer.item() else "X"
        s = pd.Series([answer.item(), predict.item(), exact], index=predict_df.columns)
        predict_df = predict_df.append(s, ignore_index=True)


# In[ ]:


fscore_df = pd.DataFrame(columns=["category", "all","precison", "recall", "fscore"])


# In[ ]:


prediction_count = collections.Counter(predict_df["predict"])
answer_count = collections.Counter(predict_df["answer"])


# In[ ]:


for i in range(9):
    all_count = answer_count[i]
    precision = len(predict_df.query('predict == ' + str(i) + ' and exact == "O"')) / prediction_count[i]
    recall = len(predict_df.query('answer == ' + str(i) + ' and exact == "O"')) / all_count
    fscore = 2*precision*recall / (precision + recall)
    s = pd.Series([index2category[i], all_count, round(precision, 2), round(recall, 2), round(fscore, 2)], index=fscore_df.columns)
    fscore_df = fscore_df.append(s, ignore_index=True)
print(fscore_df)


# In[ ]:




