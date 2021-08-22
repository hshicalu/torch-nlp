# PyTorchを使って自然言語処理の勉強をする
Qiitaの記事を参考にします
  
### LSTMからBERTまでを学ぶ  

| タイトル | チェック |
|---- | :----: |
| [PyTorchを使ってLSTMで文章分類を実装してみた](https://qiita.com/m__k/items/841950a57a0d7ff05506) |<li>[x] </li>|
| [PyTorchを使ってLSTMで文章分類を実装してみた（バッチ化対応ver）](https://qiita.com/m__k/items/db1a81bb06607d5b0ec5) |<li>[x] </li>|
| [PyTorchでSeq2Seqを実装してみた](https://qiita.com/m__k/items/b18756628575b177b545) |<li>[ ] </li>|
| [PyTorchでAttention Seq2Seqを実装してみた](https://qiita.com/m__k/items/646044788c5f94eadc8d) |<li>[ ] </li>|
| [PyTorchのBidirectional LSTMのoutputの仕様を確認してみた](https://qiita.com/m__k/items/78a5125d719951ca98d3) |<li>[ ] </li>|
| [PyTorchでSelf Attentionによる文章分類を実装してみた](https://qiita.com/m__k/items/98ff5fb4a4cb4e1eba26) |<li>[ ] </li>|
| [PyTorchで日本語BERTによる文章分類＆Attentionの可視化を実装してみた](https://qiita.com/m__k/items/e312ddcf9a3d0ea64d72) |<li>[ ] </li>|  

[@m_kさん](https://qiita.com/m__k)に感謝  

### tips
1. datasetを作るのはColabでは時間がかかるので、予めCSVとして保存しておく  
詳しくは、`datasets.py`を参照    
2. 保存したCSVの読み込みと必要な変数の定義
```
datasets = pd.read_csv("/content/drive/MyDrive/...{カスタマイズ}.../datasets.csv")
categories = set(datasets["category"])
```
3. Colabでドライブをマウントするおまじない
```
from google.colab import drive
drive.mount('/content/drive')
```

### notionの学習メモ
https://hicalu.notion.site/PyTorch-0461e9b277e74ebeb2bd16af165ae328
