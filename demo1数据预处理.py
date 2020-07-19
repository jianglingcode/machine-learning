#coding:utf-8
import os
import csv
from csv import reader
from csv import writer
import pandas as pd
import numpy as np
import re
import jieba
import gensim
from gensim.models import word2vec
import matplotlib.pyplot as plt 
from wordcloud import WordCloud

# dirpath=os.getcwd()
stop = pd.read_csv("stoplist.txt",encoding="utf-8",header=None,sep="tipdm",engine="python")
stop = list(stop[0])

with open('text.csv','r', encoding='utf-8') as f:
    data=f.read()

jieba.load_userdict('dict.txt')

def del_stopwords(data):
    stopwords=stop
    for word in stopwords:
        if word in data:
            data=data.replace(word,'')
    return data

#评论整体相同去重
def file_remove(data):
    new_data=list(set(data))
    sub=len(data)-len(new_data)
    print('删去了{}条评论'.format(sub))
    return new_data

#机械压缩,去短句
def jixie(data):
    newdata=[]
    for line in data:
        for j in range(len(line)):       #一个字重复至少三次 删除
            if line[j:j+1]==line[j+1:j+2] and line[j:j+1]==line[j+2:j+3]:
                k=j+2
                while line[k:k+1]==line[k+1:k+2] and k<len(line):
                    k=k+1
                line=line[:j]+line[k:]
        for i in range(2,5):             #多个字至少重复两次 删除
            for j in range(len(line)):
                if line[j:j+i]==line[j+i:j+i+i]:
                    k=j+i
                    while line[k:k+i]==line[k+i:k+i+i] and k<len(line):
                        k=k+i
                    line=line[:j]+line[k:]
        # print(line)


        #去短句
        if len(line)>4 :
            newdata.append(line)
    return newdata

#re模块去除标点符号
def remove(data):
    newdata=[]
    punctuation = '，！。：.（）()；？;:?、",^&~*!#%\'+-/ ' #此处有空格
    for line in data:
        line = re.sub(r'[%s]+'%punctuation, '', line)
        line=re.sub('[0-9a-zA-Z]','',line)
        newdata.append(line)
    return newdata

#获取结巴分词
def get_jieba_words(data):
    model = gensim.models.Word2Vec.load("word2vec_atec")
    all_aim_word={}
    while True:
        words_after_jieba = [[word for  word  in  jieba.cut(line) if word.strip()] for  line  in data]   
        new_words = []
        for  line  in words_after_jieba:
            for word  in  line:
                # print(word)
                if word not in model.wv.vocab  and word not in all_aim_word.keys():
                    all_aim_word [word] = 1
                    new_words.append(word)
                elif word not in model.wv.vocab:
                    all_aim_word [word] += 1
                    
        if  len(new_words) < 10:
            for word in new_words:
                # print('new_word',word)
                pass
            for word in  all_aim_word:
                if all_aim_word[word]>5:   
                    # print(word, all_aim_word[word])
                    pass
            return  words_after_jieba
        else:
            for  word  in new_words:
                jieba.del_word(word)
    raise Exception('提取结巴分词数据失败')


#删去低频词
def  remove_words(data):
    newdata=[]
    model = gensim.models.Word2Vec.load("word2vec_atec")

    for line in data:
        for word in line:
            if word not in model.wv.vocab:
                line.remove(word)
        newdata.append(line)
    return newdata

#汇总得到关键词，绘制词云
def key_words(data):
    segments=[]
    for line in data:
        # print(line)
        for word in line:
            # print(word)
            segments.append({'word':word,'count':1})
    dfwords=pd.DataFrame(segments)
    dfsort=dfwords.groupby('word')['count'].sum()
    dfsort=dfsort.sort_values(ascending=False)
    print(dfsort.head(50))

    #绘制词云
    wordcloud=WordCloud(font_path='STKAITI.TTF',width=800,height=600,max_words=100,mode='RGBA',background_color=None)
    my_wordcloud=wordcloud.fit_words(dfsort)
    plt.imshow(my_wordcloud)
    plt.axis('off')
    plt.show()

data=del_stopwords(data)
data=data.split('\n')
newdata=file_remove(data)
newdata2=jixie(newdata)
newdata3=remove(newdata2)
words_after_jieba=get_jieba_words(newdata3)#二维列表
remove_w=remove_words(words_after_jieba)
key_words(remove_w)

# 写入csv
with open('text2.csv','a',encoding='utf8',newline='')as f6:
    writer=csv.writer(f6)
    for line in remove_w:
        writer.writerow(line)
print("写入完毕")

#重新写入txt中,写入的编码为ansi
with open('text2.csv','r',encoding='utf-8')as f:
    read=f.readlines()
    with open('text3.txt','a') as ff:
        for line in read:
            ff.writelines(line)