import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 讀取資料集
data = pd.read_csv('labeled_tweets.csv')
data.dropna(inplace=True) 

# 將文本數據轉換為數值型特徵
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])

# 將資料集拆分為訓練集和測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, data['topic'], test_size=0.2, random_state=42)



# 訓練模型
clf = MultinomialNB()
clf.fit(X_train, y_train)

# 預測測試集
y_pred = clf.predict(X_test)

# 評估分類器
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)