#-*- coding:utf-8 -*-

import pandas as pd
import os

#引入文本，初始化参数
negfile = 'text3_负面情感结果.txt'
posfile = 'text3_正面情感结果.txt'

neg = pd.read_csv(negfile,encoding="UTF-16LE",header=None,sep='\t')
neg.columns=['num',0]
pos = pd.read_csv(posfile,encoding="UTF-16LE",header=None,sep='\t')
pos.columns=['num',0]

neg[1] = neg[0].apply(lambda s:s.strip().split(" "))
pos[1] = pos[0].apply(lambda s:s.strip().split(" "))


from gensim import corpora,models

#制作词典:词语的向量化
neg_dict = corpora.Dictionary(neg[1])
pos_dict = corpora.Dictionary(pos[1])

#建立语料库
neg_corpus = [neg_dict.doc2bow(i) for i in neg[1]]
pos_corpus = [pos_dict.doc2bow(i) for i in pos[1]]

#LDA模型训练
neg_lda = models.LdaModel(neg_corpus,num_topics=4,id2word=neg_dict)  
pos_lda = models.LdaModel(pos_corpus,num_topics=4,id2word=pos_dict)

#打印输出主题
print('输出负面情感：')
print(neg_lda.print_topics(num_topics=3))
f = open("neg.txt","w",encoding="utf-8")
for i in range(3):
  f.write(neg_lda.print_topic(i))
  f.write("\n")

print('输出正面情感：')
print(pos_lda.print_topics(num_topics=3))
f = open("pos.txt","w",encoding="utf-8")
for i in range(3):
  f.write(pos_lda.print_topic(i))
  f.write("\n")


import pyLDAvis
import gensim
from gensim import models
import pyLDAvis.gensim

#正面
# vis = pyLDAvis.gensim.prepare(pos_lda,pos_corpus,pos_dict)
# pyLDAvis.show(vis)

#负面
vis2 = pyLDAvis.gensim.prepare(neg_lda,neg_corpus,neg_dict)
pyLDAvis.show(vis2)