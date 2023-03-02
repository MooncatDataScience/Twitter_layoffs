import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
nltk.data.path.insert(0, "F:\\nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


df = pd.read_csv('labeled_tweets.csv', encoding='utf-8')
df = df[['text', 'topic']]
df.dropna(inplace=True) 

# 分詞和停用
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#添加特殊字符 [CLS] 和 [SEP]，
df['tokens'] = df['text'].apply(lambda x: [w.lower() for w in tokenizer.tokenize(str(x)) if w.lower() not in stop_words])
df['input_ids'] = df['tokens'].apply(lambda x: bert_tokenizer.convert_tokens_to_ids(x))

# 計算最大長度並補齊
MAX_LEN = max([len(x) for x in df['input_ids']])
df['input_ids'] = df['input_ids'].apply(lambda x: x + [bert_tokenizer.pad_token_id] * (MAX_LEN - len(x)))

# 1表示真實的輸入，0表示填充的部分
df['attention_mask'] = df['input_ids'].apply(lambda x: [int(token_id != bert_tokenizer.pad_token_id) for token_id in x])



# 使用OneHotEncoder將整數字編碼轉換為二次製造向量的形式
le = LabelEncoder()
df['label'] = le.fit_transform(df['topic'])
ohe = OneHotEncoder()
labels = ohe.fit_transform(df['label'].values.reshape(-1, 1)).toarray()

train = np.array(df['attention_mask'])
test = np.array(labels)

# 載入 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# 將 train 和 test 轉換為 BERT 的格式
train_inputs = []
test_inputs = []

for i in range(len(train)):
    # 將每一個樣本轉換為 BERT 的格式
    input_ids = train[i]
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)
    train_inputs.append({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    })

for i in range(len(test)):
    # 將每一個樣本轉換為 BERT 的格式
    input_ids = test[i]
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)
    test_inputs.append({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    })

# 訓練集佔 80%，測試集佔 20%
train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, test, 
                                                    random_state=2022, test_size=0.2)

# 將訓練集和測試集轉換為 NumPy 陣列
train_inputs = np.array(train_inputs)
val_inputs = np.array(val_inputs)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

print(train_inputs.shape)
print(val_inputs.shape)
print(train_labels.shape)
print(val_labels.shape)

# 定義一個函數，將特徵和標籤打包成 Dataset
def create_dataset(inputs, labels, batch_size):
    def gen():
        for i in range(len(inputs)):
            yield (inputs[i], labels[i])
    
    dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.float32, tf.float32),
        (tf.TensorShape([MAX_LEN]), tf.TensorShape([len(labels[0])]))
    )
    dataset = dataset.batch(batch_size)
    return dataset

# 設定超參數
BATCH_SIZE = 32

# 創建訓練集和測試集的 Dataset
train_dataset = create_dataset(train_inputs, train_labels, BATCH_SIZE)
val_dataset = create_dataset(val_inputs, val_labels, BATCH_SIZE)



'''
未完成 : 訓練模型、驗證模型、預測分類
'''