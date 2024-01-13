#!/usr/bin/env python
# coding: utf-8

# 来源：https://blog.csdn.net/weixin_42617035/article/details/102680583  <br>
# https://blog.csdn.net/google19890102/article/details/25592833 （贝叶斯）假设特征相互独立 <br>
# https://blog.csdn.net/google19890102/article/details/80091502 （snownlp原理讲解） <br>
# https://blog.csdn.net/weixin_41961559/article/details/105237852 (推荐了工具)<br>

# ### 导入数据

# In[40]:


import numpy as np
import pandas as pd


# In[41]:


data = pd.read_excel('data_test_train.xlsx')
print(data.head())


# ### 朴素贝叶斯

# #### 数据预处理 

# In[42]:


#根据需要做处理
#去重、去除停用词


# #### jieba分词

# In[43]:


import jieba

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

data['cut_comment'] = data.comment.apply(chinese_word_cut)


# In[44]:


data.head()


# #### 提取特征

# In[45]:


from sklearn.feature_extraction.text import CountVectorizer

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

stop_words_file = '哈工大停用词表.txt'
stopwords = get_custom_stopwords(stop_words_file)

vect = CountVectorizer(max_df = 0.8, 
                       min_df = 3, 
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b', 
                       stop_words=list(stopwords))


# #### 划分数据集

# In[46]:


#划分数据集
X = data['cut_comment']
y = data.sentiment

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)


# In[47]:
# print(vect.get_feature_names_out())

#特征展示
test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names_out())
print(test.head())


# #### 训练模型

# In[48]:


from sklearn.naive_bayes import MultinomialNB#朴素贝叶斯分类器
nb = MultinomialNB()

X_train_vect = vect.fit_transform(X_train)
nb.fit(X_train_vect, y_train)
train_score = nb.score(X_train_vect, y_train)
print(train_score)


# #### 测试模型

# In[49]:


X_test_vect = vect.transform(X_test)
print(nb.score(X_test_vect, y_test))


# #### 分析数据 

# In[50]:


data = pd.read_excel("data.xlsx")
print(data.head())



# In[51]:


data = pd.read_excel("data.xlsx")
data['cut_comment'] = data.comment.apply(chinese_word_cut)
X=data['cut_comment']
X_vec = vect.transform(X)
nb_result = nb.predict(X_vec)
#predict_proba(X)[source] 返回概率
data['nb_result'] = nb_result


# In[52]:


test = pd.DataFrame(vect.fit_transform(X).toarray(), columns=vect.get_feature_names_out())
print(test.head())



# In[53]:


data.to_excel("data_result.xlsx",index=False)


# ### snownlp(了解即可)

# 这里的snownlp没有做训练，效果不好，所以用贝叶斯的方法就行。如果要使用snownlp还是要训练一下。

# In[10]:


from snownlp import SnowNLP

text1 = '这个东西不错'
text2 = '这个东西很垃圾'

s1 = SnowNLP(text1)
s2 = SnowNLP(text2)

print(s1.sentiments,s2.sentiments)


# In[ ]:


def snow_result(comemnt):
    s = SnowNLP(comemnt)
    if s.sentiments >= 0.6:
        return 1
    else:
        return 0

data['snlp_result'] = data.comment.apply(snow_result)

data.head(5)


# In[ ]:


counts = 0
for i in range(len(data)):
    if data.iloc[i,2] == data.iloc[i,3]:
        counts+=1

print(counts/len(data))

