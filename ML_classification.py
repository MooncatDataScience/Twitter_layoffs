import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 讀取資料集
data = pd.read_csv('labeled_tweets.csv')
data.dropna(inplace=True) 

vectorizers = [
    CountVectorizer(stop_words='english'),
    TfidfVectorizer(stop_words='english')
]

classifiers = [
    MultinomialNB(),
    LogisticRegression(),
    LinearSVC(),
    RandomForestClassifier(n_estimators=100, random_state=42)
]

results = []
for vectorizer in vectorizers:
    # 將文本數據轉換為數值型特徵
    X = vectorizer.fit_transform(data['text'].astype('U'))

    # 將資料集拆分為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, data['topic'], test_size=0.2, random_state=42)

    for classifier in classifiers:
        # 訓練模型
        clf = classifier
        clf.fit(X_train, y_train)

        # 預測測試集
        y_pred = clf.predict(X_test)

        # 評估分類器
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        result = {
            'Vectorizer': vectorizer.__class__.__name__,
            'Classifier': classifier.__class__.__name__,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 score': f1
        }
        results.append(result)

df_results = pd.DataFrame(results)
print(df_results)
