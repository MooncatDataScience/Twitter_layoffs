# 文本分類實驗
在這個實驗中，我們使用 Kaggle 上的 **Twitter Layoffs** 資料集作為文本分類的示例。

我們使用 Python 和 Scikit-learn 模組進行文本分類，運用 CountVectorizer 和 TfidfVectorizer 進行文本特徵提取，使用多個分類器進行訓練和預測，並比較了所有分類器的性能表現。

### 實驗方法
1. 讀取資料集，並使用 Pandas 對資料進行了填充缺失值和將分類標籤轉換為數字的處理。
2. 選擇 CountVectorizer 作為特徵提取器，並將文本數據轉換為數值型特徵。
3. 使用 train_test_split 函數將資料集拆分為訓練集和測試集。
4. 使用多個分類器進行了訓練和預測，包括 Naive Bayes、Logistic Regression、SVM、Decision Tree 和 Random Forest。
5. 使用 XGBoost 分類器進行了訓練和預測，並比較了所有分類器的性能表現，使用 accuracy、precision、recall 和 F1 score 作為評估指標。

### 結論
在本次實驗中，我們發現 CountVectorizer 特徵提取器與 LogisticRegression 分類器結合的模型具有較好的性能表現，取得了最高的準確率和 F1 score。

因此，在實際應用中，我們可以根據文本數據的特性選擇最適合的特徵提取器和分類器，以達到最佳的文本分類效果。

Vectorizer              Classifier  Accuracy  Precision    Recall  F1 score
0  CountVectorizer           MultinomialNB  0.609107   0.651842  0.609107  0.585572
1  CountVectorizer      LogisticRegression  0.720790   0.719501  0.720790  0.713198
2  CountVectorizer               LinearSVC  0.716495   0.712330  0.716495  0.710982
3  CountVectorizer  RandomForestClassifier  0.593643   0.598203  0.593643  0.562949
4  TfidfVectorizer           MultinomialNB  0.505584   0.685638  0.505584  0.436365
5  TfidfVectorizer      LogisticRegression  0.692010   0.703428  0.692010  0.672345
6  TfidfVectorizer               LinearSVC  0.737973   0.735184  0.737973  0.730016
7  TfidfVectorizer  RandomForestClassifier  0.597938   0.632584  0.597938  0.554677

---

EDA_Analysis.ipynb : 做EDA分析

tes.ipynb :　寫一些測試

labeled_tweets.csv　: 文本特徵

tweets.csv : 初始csv

ML_classification.py : 使用貝氏分類

ML_labeled_tweets : 使用LDA將文本分成1~5的分類

ML_xgboost :　使用xgboost

NLTK : 使用NLTK+bert來訓練(尚未完成)

