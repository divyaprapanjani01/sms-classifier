#!/usr/bin/env python
# coding: utf-8

# In[51]:


pip install pandas


# In[52]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[65]:


# Assuming your dataset has a 'v1' column containing labels and a 'v2' column containing SMS messages
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
# Convert 'label' values to numerical values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)


# In[67]:


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# In[68]:


model = MultinomialNB()
model.fit(X_train_vec, y_train)


# In[69]:


y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[70]:


new_sms = ["Free gift! Click now!", "Hello, how are you?"]
new_sms_vec = vectorizer.transform(new_sms)
predictions = model.predict(new_sms_vec)

for sms, prediction in zip(new_sms, predictions):
    print(f"{sms} - {'Spam' if prediction == 1 else 'Non-Spam'}")


# In[ ]:




