import pandas as pd 
import numpy as np
import jieba
import gensim
from gensim.models import word2vec
from sklearn.model_selection import train_test_split


neg_content=pd.read_csv('text3_负面情感结果.txt',header=None,sep='\t',encoding='UTF-16LE')
neg_content.columns=['num','text']
neg_content['mark']=0
print(neg_content.head())

pos_content=pd.read_csv('text3_正面情感结果.txt',header=None,sep='\t',encoding='UTF-16LE')
pos_content.columns=['num','text']
pos_content['mark']=1


content=pd.concat([neg_content,pos_content],axis=0,ignore_index=True)
content['text'] = content['text'].apply(lambda x:" ".join(str(x).split(' ')))
print(content.head())

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split

#打乱
x_train, x_val, y_train, y_val = train_test_split(content['text'],content['mark'], test_size=0.4)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(content['text']))
 
#生成token词典
train_x =  tokenizer.texts_to_sequences(x_train)
test_x = tokenizer.texts_to_sequences(x_val)
  
#转换为word下标的向量形式
maxlen = 30
train_x = pad_sequences(train_x, padding='post', maxlen=maxlen)
test_x = pad_sequences(test_x, padding='post', maxlen=maxlen)
 
embedding_dim = 50
vocab_size = len(tokenizer.word_index) +1 
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                          output_dim=embedding_dim,
                          input_length=maxlen))
model.add(layers.Flatten())
#layers.Dense：全连接层。在整个卷积神经网络中起到“分类器”的作用
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))
#model.compile()：将优化器传递给之前实例化优化器
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.summary()

# to_categorical：将类向量（整数）转换为二进制类矩阵，神经网络只能fit(训练)矩阵，用来将整数型标签转化为one_hot数据
model.fit(train_x[:400], to_categorical(y_train.astype(int))[:400],
         epochs=1,
         batch_size=10)
test_sub = model.predict_classes(test_x)
print(test_sub)
score = model.evaluate(test_x[:400], to_categorical(y_val.astype(int))[:400],batch_size=10)
print(score)
