import re
import jieba
import collections
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

with open("stoplist.txt",'r',encoding='utf8')as f:
    stop=f.readlines()
    stop = [x.replace('\n','') for x in stop]

num=40  #词的个数
G=nx.Graph()
plt.figure(figsize=(20,14))
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号
plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签

# 读取csv分词后文件，以列表形式存储
def read_csv(filename):
    with open(filename,'r',encoding='utf8')as f:
        data=f.read()                                 #"word1,word2,word3\nword4,word5\n"
        data=data.replace('\n',',')
        data=data.split(',')
        
    return data

word_list=read_csv('text2.csv')

#词频统计
word_counts=collections .Counter(word_list)  #对分词做词频统计
word_counts_top=word_counts.most_common(num)  #获取最高频的词
word=pd.DataFrame(word_counts_top,columns=['关键词','次数'])
word=word.drop(index=(word.loc[word['关键词']=='不']).index)
word=word.drop(index=(word.loc[word['关键词']=='没']).index)
word=word.drop(index=(word.loc[word['关键词']=='大']).index)
word=word.drop(index=(word.loc[word['关键词']=='功']).index)
word=word.drop(index=(word.loc[word['关键词']=='价']).index)
word=word.drop(index=(word.loc[word['关键词']=='力']).index)
word=word.drop(index=(word.loc[word['关键词']=='但']).index)
word=word.drop(index=(word.loc[word['关键词']=='老']).index)
word=word.drop(index=(word.loc[word['关键词']=='拍']).index)
word=word.drop(index=(word.loc[word['关键词']=='多']).index)
word=word.drop(index=(word.loc[word['关键词']=='支持']).index)
word=word.drop(index=(word.loc[word['关键词']=='问题']).index)
word=word.drop(index=(word.loc[word['关键词']=='特别']).index)
word=word.drop(index=(word.loc[word['关键词']=='够']).index)
word=word.drop(index=(word.loc[word['关键词']=='超级']).index)
word=word.drop(index=(word.loc[word['关键词']=='特色']).index)
word=word.drop(index=(word.loc[word['关键词']=='长']).index)


print(word)
num=word.shape[0]
print(num)

# 文本预处理

#去除标点符号
#文本分词，精确模式
#去除停用词，列表存储 存储两遍word_list,list2
#去除低频词

with open('text.csv','r', encoding='utf-8') as f:
    data=f.readlines()

#评论整体相同去重
def file_remove(data):
    new_data=list(set(data))
    return new_data
newdata=file_remove(data)
print("1.整体去重，写入完毕")
#print(newdata)
#print('整体去重后：'+str(len(newdata)))

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
newdata2=jixie(newdata)
print("2.去短句，写入完毕")

#re模块去除标点符号
def remove(data):
    newdata=[]
    punctuation = '，！。：.（）()；？;:?、",^&~~ ' #此处有空格
    for line in data:
        line = re.sub(r'[%s]+'%punctuation, '', line)
        newdata.append(line)
    return newdata
newdata3=remove(newdata2)
print("3.去除标点符号，写入完毕")
newdata3=''.join(newdata3)

#做转置，形成矩阵
word_T = pd.DataFrame(word.values.T,columns=word.iloc[:,0])
# print('~~'*30)
# print(word_T)

net=pd.DataFrame(np.mat(np.zeros((num,num))),columns=word.iloc[:,0])
# print('~~'*30)
# print(net)
k=0
index=0
#构建语义关联矩阵
for i in range(len(newdata3)):
    if newdata3[i] == '\n':  #根据换行符读取一段文字
        data_exact = jieba.cut(newdata3[k:i], cut_all = False) # 精确模式分词
        word_list2 = []
        for words in data_exact: # 循环读出每个分词
            if words not in stop: # 如果不在去除词库中
                word_list2.append(words) # 分词追加到列表
        
        index+=1
        if index>1000:
            word_counts2=collections.Counter(word_list2)
            word_counts_top2=word_counts2.most_common(10) #获取该断最高频词
            word2=pd.DataFrame(word_counts_top2)
            word2_T=pd.DataFrame(word2.values.T,columns=word2.iloc[:,0])
            relation=list(0 for x in range(num))
            # 查看该段最高频词是否总在最高频的词列表中
            for j in range(num):
                for p in range(len(word2)):
                    if word.iloc[j,0]==word2.iloc[p,0]:
                        relation[j]=1
                        break
            # 对于同段落内出现的最高频词，根据其出现次数加到语义关联矩阵的相应位置
            for j in range(num):
                if relation[j]==1:
                    for q in range(num):
                        if relation[q]==1:
                            net.iloc[j,q]=net.iloc[j,q]+word2_T.loc[1,word_T.iloc[0,q]]
            index=0
        k=i+1    
#    处理最后一段内容，完成语义关联矩阵的构建

n=len(word)
#    边的起点、终点、权重
for i in range(n):
    for j in range(i,n):
        G.add_weighted_edges_from([(word.iloc[i,0],word.iloc[j,0],net.iloc[i,j])])
# print(G.edges)
nx.draw_networkx(G,
                    pos=nx.spring_layout(G),
                    edge_color='orange',
                    
                    )
plt.axis('off')
plt.show()