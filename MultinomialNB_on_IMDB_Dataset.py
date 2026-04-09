#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string 
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[2]:


data = pd.read_csv(r"C:\Users\sidde\OneDrive\Desktop\imdb dataset of 50k Movie Reviews\IMDB Dataset.csv")
data


# In[3]:


data.loc[1,'review']


# In[4]:


def clean_text_from_df(text):
  # 1. Remove URLs/Links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 2. Remove Twitter mentions (@users)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Remove punctuation (keeping only letters, numbers, and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Remove extra whitespace/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Optional: Convert to lowercase for standardization
    text = text.lower()
    return text


# In[5]:


nltk.download('punkt')
nltk.download('stopwords')


# In[6]:


def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


# In[7]:


data["clean_text"] = data["review"].apply(lambda x: preprocess_text(clean_text_from_df(x)))


# In[8]:


count = CountVectorizer()


# In[9]:


X = count.fit_transform(data['clean_text'])
Y = data['sentiment']


# In[10]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)


# In[11]:


X_train


# In[12]:


len(Y_train)


# In[13]:


model = MultinomialNB(alpha = 0.1)


# In[14]:


model.fit(X_train,Y_train)


# In[15]:


pred = model.predict(X_test)
pred


# In[16]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[17]:


import seaborn as sns


# In[18]:


score = accuracy_score(Y_test,pred)
score


# In[19]:


con = confusion_matrix(Y_test,pred)
report = classification_report(Y_test,pred)
print(report)


# In[26]:


sns.heatmap(con, annot = True,cmap = "Blues")


# In[21]:


def predict_review(text):
    # Clean + preprocess
    text = clean_text_from_df(text)
    text = preprocess_text(text)
    
    # Vectorize
    vector = count.transform([text])
    
    # Predict
    result = model.predict(vector)
    
    print("Sentiment:", result[0])


# In[22]:


predict_review("This movie was amazing and exciting")


# In[23]:


predict_review("worst movie i have ever had")


# In[24]:


predict_review("the suspense in the movie was thriller")


# In[ ]:




