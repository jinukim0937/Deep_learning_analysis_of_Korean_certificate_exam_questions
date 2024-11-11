# Deep learning analysis of Korean certificate exam questions

## 개요

2022학년도 2학기 딥러닝과 응용 수업의 프로젝트이다. 이 프로젝트의 목표는 시험 문제 데이터를 효율적으로 관리할 수 있는 딥러닝 알고리즘을 개발하는 것으로, 한국어 자격시험 데이터에 대해 서로 다른 기능을 수행하는 세 가지 모델을 개발하였다. 

사용한 모델: LSTM, KoNLPy, BERT



## 팀원

김진우, 최형호, 이학민



## 코드

# Model1: 선지 도메인 분류
## 참고사이트

https://gist.github.com/Lucia-KIM/165b8f13c007f83b4762ab436ea95610

https://byumm315.tistory.com/entry/%ED%95%9C%EA%B5%AD%EC%96%B4-%EB%89%B4%EC%8A%A4-%ED%86%A0%ED%94%BD-%EB%B6%84%EB%A5%98-1-I-am-yumida

## 필요한 패키지 설치


```python
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```


```python
!pip install -q tweepy==3.10

try:
    import konlpy
except:
    !pip install -q konlpy
    import konlpy
```

    [K     |████████████████████████████████| 19.4 MB 4.8 MB/s 
    [K     |████████████████████████████████| 465 kB 81.0 MB/s 
    [?25h

## 패키지 불러오기


```python
import pandas as pd
import numpy as np
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt

import matplotlib as mpl
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
```

## 데이터 불러오기


```python
data_raw = pd.read_excel("Dataset.xlsx", engine = "openpyxl")
```


```python
data1 = data_raw[['Testname','ChoiceText1']]
data2 = data_raw[['Testname','ChoiceText2']]
data3 = data_raw[['Testname','ChoiceText3']]
data4 = data_raw[['Testname','ChoiceText4']]
data1.columns = ['Testname','ChoiceText']
data2.columns = ['Testname','ChoiceText']
data3.columns = ['Testname','ChoiceText']
data4.columns = ['Testname','ChoiceText']
```


```python
data = pd.concat([data1,data2,data3,data4], axis=0)
```


```python
data = data.reset_index(drop=True,inplace=False)
```

## 데이터 전처리


```python
data = data.dropna()
```


```python
data.loc[(data['Testname'] == "건축기사"), 'Testname'] = 0 
data.loc[(data['Testname'] == "대기환경기사"), 'Testname'] = 1
data.loc[(data['Testname'] == "산업안전기사"), 'Testname'] = 2 
data.loc[(data['Testname'] == "소방설비기사"), 'Testname'] = 3 
data.loc[(data['Testname'] == "정보처리기사"), 'Testname'] = 4 
```


```python
def cleanText(readData):
    text = re.compile('[^ A-Za-z0-9가-힣]+')
    result = text.sub('', readData)
    return result
```


```python
for i in range(len(data)):
    data.iloc[i]['ChoiceText'] = cleanText(data.iloc[i]['ChoiceText'])
```


```python
#def isKorean(text):
#    hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')  
#    result = hangul.findall(text)
#    return len(result)
#data_fin =pd.DataFrame({'Testname':[],'ChoiceText':[]})
#for i in range(len(data)):
#    if isKorean(data.iloc[i]['ChoiceText']) != 0:
#      data_fin = data_fin.append({'Testname':data.iloc[i]['Testname'],'ChoiceText':data.iloc[i]['ChoiceText']}, ignore_index=True)
#    else:
#      continue
```


```python
data
```





  <div id="df-1ba7731b-1935-4cba-a088-b156ec03a007">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Testname</th>
      <th>ChoiceText</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>쇼룸</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>공사비가저렴하다</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>T자형</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>폐가식</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>상층침실하층침실</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13795</th>
      <td>4</td>
      <td>특정하드웨어에종속되어특화된업무를서비스하기에적합하다</td>
    </tr>
    <tr>
      <th>13796</th>
      <td>4</td>
      <td>SecureOS</td>
    </tr>
    <tr>
      <th>13797</th>
      <td>4</td>
      <td>조건이복합되어있는곳의처리를시각적으로명확히식별하는데적합하다</td>
    </tr>
    <tr>
      <th>13798</th>
      <td>4</td>
      <td>Logs</td>
    </tr>
    <tr>
      <th>13799</th>
      <td>4</td>
      <td>SPICE</td>
    </tr>
  </tbody>
</table>
<p>11776 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1ba7731b-1935-4cba-a088-b156ec03a007')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }
    
    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }
    
    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }
    
    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1ba7731b-1935-4cba-a088-b156ec03a007 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';
    
        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1ba7731b-1935-4cba-a088-b156ec03a007');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;
    
          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## train & test 데이터로 나누기


```python
x = data['ChoiceText']
y = data['Testname']
```


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=10)
```


```python
train_data = pd.DataFrame({'document':x_train,
                             'label':y_train})
```


```python
test_data = pd.DataFrame({'document':x_test,
                             'label':y_test})
```

## 워드 임베딩



Word Tokenization


```python
okt = Okt()
```


```python
X_train = []
for sentence in train_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # Okt() 이용해서 토큰화
    X_train.append(temp_X)
```


```python
X_test = []
for sentence in test_data['document']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    X_test.append(temp_X)
```

Tokenizer를 이용하여 데이터에 있는 고유한 단어에 번호 할당

-> 정수 인덱싱 벡터화

-> 모든 문장의 단어 벡터 길이는 같아야하기 때문에, 같지 않은 경우 0으로 패딩해서 길이를 맞춘다.


```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
```


```python
total_count = len(tokenizer.word_index) # 단어의 수
vocab_size = total_count + 2
print('단어 집합의 크기 :',vocab_size)
```

    단어 집합의 크기 : 8957



```python
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
```


```python
y_train = np.array(train_data['label'],dtype=np.int)
y_test = np.array(test_data['label'],dtype=np.int)
```

    <ipython-input-27-a6268166a168>:1: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      y_train = np.array(train_data['label'],dtype=np.int)
    <ipython-input-27-a6268166a168>:2: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      y_test = np.array(test_data['label'],dtype=np.int)



```python
#drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
```


```python
print('최대 길이 :', max(len(l) for l in X_train))
print('평균 길이 :', sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('길이')
plt.ylabel('수')
plt.show()
```

    최대 길이 : 61
    평균 길이 : 6.502264578222306


    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 44600 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 51060 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 49688 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 44600 missing from current font.
      font.set_text(s, 0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 51060 missing from current font.
      font.set_text(s, 0, flags=flags)
    /usr/local/lib/python3.8/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 49688 missing from current font.
      font.set_text(s, 0, flags=flags)




![png](output_34_2.png)
    



```python
max_len = 30
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
```


```python
# 패딩 작업이 잘 되었나 확인
plt.hist([len(s) for s in X_train], bins=50)
plt.show()
```


​    
![png](output_36_0.png)
​    


## 데이터 레이블 균형 맞추기 & 데이터 증강


```python
pd.Series(y_train).value_counts()
```




    0    2690
    2    2531
    4    2155
    3    1868
    1    1354
    dtype: int64




```python
smote = SMOTE(random_state=0)
x_train_over,y_train_over = smote.fit_resample(X_train,y_train)
```


```python
print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', x_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())
```

    SMOTE 적용 전 학습용 피처/레이블 데이터 세트:  (10598, 30) (10598,)
    SMOTE 적용 후 학습용 피처/레이블 데이터 세트:  (13450, 30) (13450,)
    SMOTE 적용 후 레이블 값 분포: 
     4    2690
    1    2690
    2    2690
    3    2690
    0    2690
    dtype: int64


y 데이터 LSTM 입력을 위한 더미화


```python
y_train_over_fin = pd.get_dummies(y_train_over).values
```

## LSTM 모델


```python
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```


```python
model1 = Sequential()

model1.add(Embedding(vocab_size, 30))
model1.add(LSTM(128))
#model1.add(Dropout(0.2))
#model1.add(Dense(16))
#model1.add(Dropout(0.2))
model1.add(Dense(5, activation='softmax'))
```


```python
model1.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, None, 30)          268710    
                                                                     
     lstm (LSTM)                 (None, 128)               81408     
                                                                     
     dense (Dense)               (None, 5)                 645       
                                                                     
    =================================================================
    Total params: 350,763
    Trainable params: 350,763
    Non-trainable params: 0
    _________________________________________________________________



```python
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
```


```python
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model1.fit(x_train_over, y_train_over_fin, epochs=10, callbacks=[es, mc], batch_size=20, validation_split=0.3)
```

    Epoch 1/10
    470/471 [============================>.] - ETA: 0s - loss: 1.4127 - acc: 0.3994
    Epoch 1: val_acc improved from -inf to 0.30632, saving model to best_model.h5
    471/471 [==============================] - 13s 10ms/step - loss: 1.4122 - acc: 0.3999 - val_loss: 1.7860 - val_acc: 0.3063
    Epoch 2/10
    464/471 [============================>.] - ETA: 0s - loss: 0.7679 - acc: 0.7172
    Epoch 2: val_acc improved from 0.30632 to 0.40768, saving model to best_model.h5
    471/471 [==============================] - 3s 7ms/step - loss: 0.7653 - acc: 0.7182 - val_loss: 1.9240 - val_acc: 0.4077
    Epoch 3/10
    471/471 [==============================] - ETA: 0s - loss: 0.4105 - acc: 0.8588
    Epoch 3: val_acc did not improve from 0.40768
    471/471 [==============================] - 3s 7ms/step - loss: 0.4105 - acc: 0.8588 - val_loss: 2.2011 - val_acc: 0.4000
    Epoch 4/10
    468/471 [============================>.] - ETA: 0s - loss: 0.2656 - acc: 0.9056
    Epoch 4: val_acc did not improve from 0.40768
    471/471 [==============================] - 3s 7ms/step - loss: 0.2668 - acc: 0.9055 - val_loss: 2.6441 - val_acc: 0.4072
    Epoch 5/10
    470/471 [============================>.] - ETA: 0s - loss: 0.2157 - acc: 0.9198
    Epoch 5: val_acc did not improve from 0.40768
    471/471 [==============================] - 3s 7ms/step - loss: 0.2157 - acc: 0.9198 - val_loss: 2.8441 - val_acc: 0.3990
    Epoch 5: early stopping


y_test 더미 변수화


```python
y_test_fin = pd.get_dummies(y_test).values
```


```python
print("\n 테스트 정확도: %.4f" %(model1.evaluate(X_test,y_test_fin)[1]))
```

    37/37 [==============================] - 0s 3ms/step - loss: 0.9758 - acc: 0.6944
    
     테스트 정확도: 0.6944


# Model2: 문제 도메인 분류
## 참고사이트

https://hoit1302.tistory.com/159

https://velog.io/@jiyoung/Text-ClassificationKoBERT%EB%A1%9C-%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0-%EC%BD%94%EB%93%9C

https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf

https://doheon.github.io/%EC%BD%94%EB%93%9C%EA%B5%AC%ED%98%84/nlp/ci-kobert-post/

## 필요한 패키지 설치


```python
!pip install gluonnlp pandas tqdm   
!pip install mxnet
!pip install sentencepiece==0.1.91
!pip install transformers==4.8.2
!pip install torch
```


```python
# 실행 후 필요시 런타임 재 실행
!pip install torch --upgrade
```


```python
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```


```python
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
```

## 패키지 불러오기


```python
import pandas as pd
import re

from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel

#GPU 사용 시
device = torch.device("cpu")
```

## 데이터 불러오기


```python
data_raw = pd.read_excel("Dataset.xlsx", engine = "openpyxl")
```


```python
data = data_raw[['Testname','MainText']]
```

## 데이터 전처리


```python
data.loc[(data['Testname'] == "건축기사"), 'Testname'] = 0 
data.loc[(data['Testname'] == "대기환경기사"), 'Testname'] = 1
data.loc[(data['Testname'] == "산업안전기사"), 'Testname'] = 2 
data.loc[(data['Testname'] == "소방설비기사"), 'Testname'] = 3 
data.loc[(data['Testname'] == "정보처리기사"), 'Testname'] = 4 
```

    /usr/local/lib/python3.8/dist-packages/pandas/core/indexing.py:1732: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._setitem_single_block(indexer, value, name)
    /usr/local/lib/python3.8/dist-packages/pandas/core/indexing.py:723: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      iloc._setitem_with_indexer(indexer, value, self.name)



```python
for i in range(len(data)):
  if data.iloc[i]['MainText'][:2] in ['1.','2.','3.','4.','5.','6.','7.','8.','9.']:
    data.iloc[i]['MainText'] = data.iloc[i]['MainText'][3:]
  elif data.iloc[i]['MainText'][:4] in ['100.']:
    data.iloc[i]['MainText'] = data.iloc[i]['MainText'][5:]
  else:
    data.iloc[i]['MainText'] = data.iloc[i]['MainText'][4:]
```


```python
def cleanText(readData):
    text = re.compile('[^ A-Za-z0-9가-힣]+')
    result = text.sub('', readData)
    return result
```


```python
for i in range(len(data)):
    data.iloc[i]['MainText'] = cleanText(data.iloc[i]['MainText'])
```

## train & test 데이터로 나누기


```python
data_list = []
for ques, label in zip(data['MainText'], data['Testname'])  :
    data = []   
    data.append(ques)
    data.append(str(label))

    data_list.append(data)
```


```python
from sklearn.model_selection import train_test_split

dataset_train, dataset_test = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=34)
```

## 워드 임베딩


```python
tokenizer = get_tokenizer()
bertmodel, vocab = get_pytorch_kobert_model()
```

    using cached model. /content/drive/MyDrive/DeeplearningPJ/.cache/kobert_news_wiki_ko_cased-1087f8699e.spiece
    using cached model. /content/drive/MyDrive/DeeplearningPJ/.cache/kobert_v1.zip
    using cached model. /content/drive/MyDrive/DeeplearningPJ/.cache/kobert_news_wiki_ko_cased-1087f8699e.spiece



```python
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         

    def __len__(self):
        return (len(self.labels))
```


```python
# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 30
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
```


```python
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test,0, 1, tok, vocab,  max_len, True, False)
```


```python
train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)
```

## BERT 분류 모델


```python
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=5,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), 
                              attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
```


```python
#BERT 모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
 
#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 대표적인 loss func

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
    
train_dataloader
```




    <torch.utils.data.dataloader.DataLoader at 0x7fcd855fe7c0>




```python
train_history=[]
test_history=[]
loss_history=[]
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
         
        #print(label.shape,out.shape)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), 
                                                                     train_acc / (batch_id+1)))
            train_history.append(train_acc / (batch_id+1))
            loss_history.append(loss.data.cpu().numpy())
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    #train_history.append(train_acc / (batch_id+1))
    
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    test_history.append(test_acc / (batch_id+1))
```

    <ipython-input-23-cdacb1bb14e9>:8: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
    Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`
      for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 1 batch id 1 loss 1.5615047216415405 train acc 0.265625
    epoch 1 train acc 0.2556818181818182


    <ipython-input-23-cdacb1bb14e9>:32: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
    Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`
      for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 1 test acc 0.3797159090909091



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 2 batch id 1 loss 1.443471908569336 train acc 0.359375
    epoch 2 train acc 0.5838068181818182



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 2 test acc 0.7541477272727272



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 3 batch id 1 loss 0.8668183088302612 train acc 0.6875
    epoch 3 train acc 0.8284801136363636



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 3 test acc 0.8701704545454546



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 4 batch id 1 loss 0.39868634939193726 train acc 0.921875
    epoch 4 train acc 0.8959517045454546



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 4 test acc 0.8961363636363636



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 5 batch id 1 loss 0.25436368584632874 train acc 0.9375
    epoch 5 train acc 0.9282670454545454



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 5 test acc 0.8981818181818183



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 6 batch id 1 loss 0.0987180545926094 train acc 0.953125
    epoch 6 train acc 0.953125



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 6 test acc 0.8882386363636364



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 7 batch id 1 loss 0.2048722356557846 train acc 0.96875
    epoch 7 train acc 0.9659090909090909



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 7 test acc 0.8782954545454547



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 8 batch id 1 loss 0.3102482259273529 train acc 0.9375
    epoch 8 train acc 0.9694602272727273



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 8 test acc 0.910340909090909



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 9 batch id 1 loss 0.11132310330867767 train acc 0.96875
    epoch 9 train acc 0.9754971590909091



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 9 test acc 0.9



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 10 batch id 1 loss 0.11789346486330032 train acc 0.96875
    epoch 10 train acc 0.9797585227272727



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 10 test acc 0.9170454545454546



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 11 batch id 1 loss 0.056651532649993896 train acc 0.96875
    epoch 11 train acc 0.9886363636363636



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 11 test acc 0.9075



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 12 batch id 1 loss 0.08256535977125168 train acc 0.96875
    epoch 12 train acc 0.9879261363636364



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 12 test acc 0.9348863636363636



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 13 batch id 1 loss 0.005675994325429201 train acc 1.0
    epoch 13 train acc 0.9911221590909091



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 13 test acc 0.9168181818181819



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 14 batch id 1 loss 0.019392674788832664 train acc 0.984375
    epoch 14 train acc 0.9928977272727273



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 14 test acc 0.9235227272727272



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 15 batch id 1 loss 0.0033295643515884876 train acc 1.0
    epoch 15 train acc 0.9982244318181818



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 15 test acc 0.925965909090909



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 16 batch id 1 loss 0.0021418025717139244 train acc 1.0
    epoch 16 train acc 0.9989346590909091



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 16 test acc 0.9241477272727273



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 17 batch id 1 loss 0.001734854537062347 train acc 1.0
    epoch 17 train acc 0.9992897727272727



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 17 test acc 0.9273863636363636



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 18 batch id 1 loss 0.0014177345437929034 train acc 1.0
    epoch 18 train acc 0.9985795454545454



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 18 test acc 0.9217045454545455



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 19 batch id 1 loss 0.0012130874674767256 train acc 1.0
    epoch 19 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 19 test acc 0.9227272727272727



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 20 batch id 1 loss 0.0011263698106631637 train acc 1.0
    epoch 20 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 20 test acc 0.9241477272727273



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 21 batch id 1 loss 0.0009582508355379105 train acc 1.0
    epoch 21 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 21 test acc 0.9255681818181819



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 22 batch id 1 loss 0.0009099962189793587 train acc 1.0
    epoch 22 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 22 test acc 0.9255681818181819



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 23 batch id 1 loss 0.000902039697393775 train acc 1.0
    epoch 23 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 23 test acc 0.9269886363636364



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 24 batch id 1 loss 0.0008371463627554476 train acc 1.0
    epoch 24 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 24 test acc 0.9269886363636364



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 25 batch id 1 loss 0.000774515385273844 train acc 1.0
    epoch 25 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 25 test acc 0.9269886363636364



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 26 batch id 1 loss 0.001037120120599866 train acc 1.0
    epoch 26 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 26 test acc 0.9269886363636364



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 27 batch id 1 loss 0.0008310111588798463 train acc 1.0
    epoch 27 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 27 test acc 0.9269886363636364



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 28 batch id 1 loss 0.0006870168726891279 train acc 1.0
    epoch 28 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 28 test acc 0.9269886363636364



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 29 batch id 1 loss 0.0007991354796104133 train acc 1.0
    epoch 29 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 29 test acc 0.9269886363636364



      0%|          | 0/44 [00:00<?, ?it/s]


    epoch 30 batch id 1 loss 0.0007096641347743571 train acc 1.0
    epoch 30 train acc 1.0



      0%|          | 0/11 [00:00<?, ?it/s]


    epoch 30 test acc 0.9269886363636364


# Model3: 유사 문제 추출
## 참고사이트 
https://github.com/Huffon/klue-transformers-tutorial/blob/master/sentence_transformers.ipynb

https://github.com/Huffon/klue-transformers-tutorial

https://github.com/jhgan00/ko-sentence-transformers

https://klue-benchmark.com/

## 필요한 패키지 설치

필요시 런타임 다시 시작


```python
!pip install sentence-transformers datasets
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting sentence-transformers
      Downloading sentence-transformers-2.2.2.tar.gz (85 kB)
    [K     |████████████████████████████████| 85 kB 308 kB/s 
    [?25hCollecting datasets
      Downloading datasets-2.7.1-py3-none-any.whl (451 kB)
    [K     |████████████████████████████████| 451 kB 20.1 MB/s 
    [?25hCollecting transformers<5.0.0,>=4.6.0
      Downloading transformers-4.25.1-py3-none-any.whl (5.8 MB)
    [K     |████████████████████████████████| 5.8 MB 76.4 MB/s 
    [?25hRequirement already satisfied: tqdm in /usr/local/lib/python3.8/dist-packages (from sentence-transformers) (4.64.1)
    Requirement already satisfied: torch>=1.6.0 in /usr/local/lib/python3.8/dist-packages (from sentence-transformers) (1.13.0+cu116)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.8/dist-packages (from sentence-transformers) (0.14.0+cu116)
    Requirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from sentence-transformers) (1.21.6)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.8/dist-packages (from sentence-transformers) (1.0.2)
    Requirement already satisfied: scipy in /usr/local/lib/python3.8/dist-packages (from sentence-transformers) (1.7.3)
    Requirement already satisfied: nltk in /usr/local/lib/python3.8/dist-packages (from sentence-transformers) (3.7)
    Collecting sentencepiece
      Downloading sentencepiece-0.1.97-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [K     |████████████████████████████████| 1.3 MB 83.5 MB/s 
    [?25hCollecting huggingface-hub>=0.4.0
      Downloading huggingface_hub-0.11.1-py3-none-any.whl (182 kB)
    [K     |████████████████████████████████| 182 kB 99.0 MB/s 
    [?25hRequirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.8/dist-packages (from huggingface-hub>=0.4.0->sentence-transformers) (6.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.8/dist-packages (from huggingface-hub>=0.4.0->sentence-transformers) (2.23.0)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.8/dist-packages (from huggingface-hub>=0.4.0->sentence-transformers) (21.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.8/dist-packages (from huggingface-hub>=0.4.0->sentence-transformers) (4.4.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.8/dist-packages (from huggingface-hub>=0.4.0->sentence-transformers) (3.8.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.8/dist-packages (from packaging>=20.9->huggingface-hub>=0.4.0->sentence-transformers) (3.0.9)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.8/dist-packages (from transformers<5.0.0,>=4.6.0->sentence-transformers) (2022.6.2)
    Collecting tokenizers!=0.11.3,<0.14,>=0.11.1
      Downloading tokenizers-0.13.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (7.6 MB)
    [K     |████████████████████████████████| 7.6 MB 79.5 MB/s 
    [?25hRequirement already satisfied: aiohttp in /usr/local/lib/python3.8/dist-packages (from datasets) (3.8.3)
    Collecting multiprocess
      Downloading multiprocess-0.70.14-py38-none-any.whl (132 kB)
    [K     |████████████████████████████████| 132 kB 99.8 MB/s 
    [?25hRequirement already satisfied: pandas in /usr/local/lib/python3.8/dist-packages (from datasets) (1.3.5)
    Collecting xxhash
      Downloading xxhash-3.1.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (212 kB)
    [K     |████████████████████████████████| 212 kB 102.7 MB/s 
    [?25hCollecting responses<0.19
      Downloading responses-0.18.0-py3-none-any.whl (38 kB)
    Requirement already satisfied: dill<0.3.7 in /usr/local/lib/python3.8/dist-packages (from datasets) (0.3.6)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /usr/local/lib/python3.8/dist-packages (from datasets) (2022.11.0)
    Requirement already satisfied: pyarrow>=6.0.0 in /usr/local/lib/python3.8/dist-packages (from datasets) (9.0.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (1.3.3)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (4.0.2)
    Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (2.1.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (6.0.3)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (22.1.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.8/dist-packages (from aiohttp->datasets) (1.8.2)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.8/dist-packages (from requests->huggingface-hub>=0.4.0->sentence-transformers) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests->huggingface-hub>=0.4.0->sentence-transformers) (2022.9.24)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests->huggingface-hub>=0.4.0->sentence-transformers) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests->huggingface-hub>=0.4.0->sentence-transformers) (1.24.3)
    Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1
      Downloading urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
    [K     |████████████████████████████████| 127 kB 100.4 MB/s 
    [?25hRequirement already satisfied: joblib in /usr/local/lib/python3.8/dist-packages (from nltk->sentence-transformers) (1.2.0)
    Requirement already satisfied: click in /usr/local/lib/python3.8/dist-packages (from nltk->sentence-transformers) (7.1.2)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.8/dist-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.8/dist-packages (from pandas->datasets) (2022.6)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.8/dist-packages (from python-dateutil>=2.7.3->pandas->datasets) (1.15.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.8/dist-packages (from scikit-learn->sentence-transformers) (3.1.0)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.8/dist-packages (from torchvision->sentence-transformers) (7.1.2)
    Building wheels for collected packages: sentence-transformers
      Building wheel for sentence-transformers (setup.py) ... [?25l[?25hdone
      Created wheel for sentence-transformers: filename=sentence_transformers-2.2.2-py3-none-any.whl size=125938 sha256=ffa35c4bcf5965274fe363e0e6276c37cf9890edaaeea0605d52ada287839fda
      Stored in directory: /root/.cache/pip/wheels/5e/6f/8c/d88aec621f3f542d26fac0342bef5e693335d125f4e54aeffe
    Successfully built sentence-transformers
    Installing collected packages: urllib3, tokenizers, huggingface-hub, xxhash, transformers, sentencepiece, responses, multiprocess, sentence-transformers, datasets
      Attempting uninstall: urllib3
        Found existing installation: urllib3 1.24.3
        Uninstalling urllib3-1.24.3:
          Successfully uninstalled urllib3-1.24.3
    Successfully installed datasets-2.7.1 huggingface-hub-0.11.1 multiprocess-0.70.14 responses-0.18.0 sentence-transformers-2.2.2 sentencepiece-0.1.97 tokenizers-0.13.2 transformers-4.25.1 urllib3-1.25.11 xxhash-3.1.0


## 패키지 불러오기


```python
import pandas as pd
import numpy as np

import math
import logging
from datetime import datetime
import re

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
```

시간 체크를 위한 logger 초기화


```python
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
```

## 사전학습 모델과 임베딩 모델 불러오기


```python
model_name = "klue/roberta-base"
```


```python
embedding_model = models.Transformer(model_name)
```


    Downloading:   0%|          | 0.00/546 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/443M [00:00<?, ?B/s]


    Some weights of the model checkpoint at klue/roberta-base were not used when initializing RobertaModel: ['lm_head.layer_norm.bias', 'lm_head.decoder.bias', 'lm_head.dense.bias', 'lm_head.decoder.weight', 'lm_head.dense.weight', 'lm_head.bias', 'lm_head.layer_norm.weight']
    - This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of RobertaModel were not initialized from the model checkpoint at klue/roberta-base and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



    Downloading:   0%|          | 0.00/375 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/248k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/752k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/173 [00:00<?, ?B/s]



```python
pooler = models.Pooling(
    embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)
```


```python
model = SentenceTransformer(modules=[embedding_model, pooler])
```

## 데이터셋 불러오기


```python
datasets = load_dataset("klue", "sts")
```


```python
testsets = load_dataset("kor_nlu", "sts")
```


```python
train_samples = []
dev_samples = []
test_samples = []

# KLUE STS 내 훈련, 검증 데이터 예제 변환
for phase in ["train", "validation"]:
    examples = datasets[phase]

    for example in examples:
        score = float(example["labels"]["label"]) / 5.0  # 0.0 ~ 1.0 스케일로 유사도 정규화

        inp_example = InputExample(
            texts=[example["sentence1"], example["sentence2"]], 
            label=score,
        )

        if phase == "validation":
            dev_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

# KorSTS 내 테스트 데이터 예제 변환
for example in testsets["test"]:
    score = float(example["score"]) / 5.0

    if example["sentence1"] and example["sentence2"]:
        inp_example = InputExample(
            texts=[example["sentence1"], example["sentence2"]],
            label=score,
        )

    test_samples.append(inp_example)
```

## 모델 학습


```python
train_batch_size = 32
```


```python
train_dataloader = DataLoader(
    train_samples,
    shuffle=True,
    batch_size=train_batch_size,
)
train_loss = losses.CosineSimilarityLoss(model=model)
```


```python
num_epochs = 10
```


```python
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples,
    name="sts-dev",
)
```


```python
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1)  # 10% of train data for warm-up
logging.info(f"Warmup-steps: {warmup_steps}")
```


```python
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
)
```


```python
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
```


```python
test_evaluator(model)
```




    0.7646286448090985



## 유사 문제 추출


```python
data = pd.read_excel("Dataset.xlsx", engine = "openpyxl")
```


```python
def cleanText(readData):
    text = re.compile('[^ A-Za-z0-9가-힣]+')
    result = text.sub('', readData)
    return result
```


```python
for i in range(len(data)):
    data.iloc[i]['MainText'] = cleanText(data.iloc[i]['MainText'])
```


```python
for i in range(len(data)):
  if data.iloc[i]['MainText'][:2] in ['1.','2.','3.','4.','5.','6.','7.','8.','9.']:
    data.iloc[i]['MainText'] = data.iloc[i]['MainText'][3:]
  elif data.iloc[i]['MainText'][:4] in ['100.']:
    data.iloc[i]['MainText'] = data.iloc[i]['MainText'][5:]
  else:
    data.iloc[i]['MainText'] = data.iloc[i]['MainText'][4:]
```


```python
corpus = data['MainText'].values.tolist()
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
queries = data['MainText'].sample(n=7).values.tolist()
```


```python
top_k = 5
for query in queries:
 query_embedding = model.encode(query, convert_to_tensor=True)
 cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
 cos_scores = cos_scores.cpu()

 #We use np.argpartition, to only partially sort the top_k results
 top_results = np.argpartition(-cos_scores, range(top_k+1))[0:top_k+1]

 print("\n\n======================\n\n")
 print("Query:", query)
 print("\nTop 5 most similar sentences in corpus:")

 for idx in top_results[1:top_k+1]:
  print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
```


​    
    ======================


​    
    Query: 소프트웨어 품질 목표 중 하나 이상의 하드웨어 환경에서 운용되기 위해 쉽게 수정될 수 있는 시스템 능력을 의미하는 것은?
    
    Top 5 most similar sentences in corpus:
    소프트웨어 품질 목표 중 주어진 시간동안 주어진 기능을 오류없이 수행하는 정도를 나타내는 것은? (Score: 0.5699)
    소프트웨어의 일부분을 다른 시스템에서 사용할 수 있는 정도를 의미하는 것은? (Score: 0.5627)
    소프트웨어 품질목표 중 쉽게 배우고 사용할 수 있는 정도를 나타내는 것은? (Score: 0.5609)
    소프트웨어 개발 표준 중 소프트웨어 품질 및 생산성 향상을 위해 소프트웨어 프로세스를 평가 및 개선하는 국제 표준은? (Score: 0.5475)
    분산 컴퓨팅 환경에서 서로 다른 기종 간의 하드웨어나 프로토콜, 통신환경 등을 연결하여 응용프로그램과 운영환경 간에 원만한 통신이 이루어질 수 있게 서비스를 제공하는 소프트웨어는? (Score: 0.4967)


​    
    ======================


​    
    Query: 달성가치(Earned Value)를 기준으로 원가관리를 시행할 때, 실제투입원가와 계획된 일정에 근거한 진행성과 차이를 의미하는 용어는?
    
    Top 5 most similar sentences in corpus:
    S/W 각 기능의 원시 코드 라인수의 비관치, 낙관치, 기대치를 측정하여 예측치를 구하고 이를 이용하여 비용을 산정하는 기법은? (Score: 0.4516)
    인간공학 연구방법 중 실제의 제품이나 시스템이 추구하는 특성 및 수준이 달성되는지를 비교하고 분석하는 연구는? (Score: 0.4514)
    물리적인 사물과 컴퓨터에 동일하게 표현되는 가상의 모델로 실제 물리적인 자산 대신 소프트웨어로 가상화함으로써 실제 자산의 특성에 대한 정확한 정보를 얻을 수 있고, 자산 최적화, 돌발사고 최소화, 생산성 증가 등 설계부터 제조, 서비스에 이르는 모든 과정의 효율성을 향상시킬 수 있는 모델은? (Score: 0.4241)
    애플리케이션의 처리량, 응답시간, 경과시간, 자원사용률에 대해 가상의 사용자를 생성하고 테스트를 수행함으로써 성능 목표를 달성하였는지를 확인하는 테스트 자동화 도구는? (Score: 0.4227)
    소프트웨어 비용 추정모형(estimation models)이 아닌 것은? (Score: 0.4140)


​    
    ======================


​    
    Query: 도시 대기오염물질의 광화학반응에 관한 설명으로 옳지 않은 것은?
    
    Top 5 most similar sentences in corpus:
    광화학적 산화제와 2차 대기오염물질에 관한 설명으로 옳지 않은 것은? (Score: 0.7382)
    광화학반응으로 생성되는 오염물질에 해당하지 않는 것은? (Score: 0.6509)
    자외선/가시선 분광법에 의한 불소화합물 분석방법에 관한 설명으로 옳지 않은 것은? (Score: 0.6305)
    흡광차분광법에 관한 설명으로 옳지 않은 것은? (Score: 0.6252)
    광화학오시던트 중 PAN에 관한 설명으로 옳은 것은? (Score: 0.6233)


​    
    ======================


​    
    Query: 대기환경보전법령상 위임업무의 보고 횟수 기준이 '수시'인 업무내용은?
    
    Top 5 most similar sentences in corpus:
    대기환경보전법규상 한국환경공단이 환경부장관에게 행하는 위탁업무 보고사항 중 “자동차배출가스 인증생략 현황”의 보고 횟수 기준은? (Score: 0.4993)
    대기환경보전법령상 위임업무 보고사항 중 “자동차 연료 및 첨가제의 제조·판매 또는 사용에 대한 규제현황” 업무의 보고횟수 기준은? (Score: 0.4878)
    대기환경보전법령상 위임업무 보고사항 중 자동차 연료 및 첨가제의 제조·판매 또는 사용에 대한 규제현황에 대한 보고횟수 기준은? (Score: 0.4811)
    대기환경보전법령상 환국환경공단이 환경부 장관에게 보고하여야 하는 위탁업무 보고사항 중 “결함확인검사 결과”의 보고기일 기준은? (Score: 0.4438)
    대기환경보전법령상 일일 기준초과배출량 및 일일유량의 산정방법으로 옳지 않은 것은? (Score: 0.4119)


​    
    ======================


​    
    Query: 기업체가 자사제품의 홍보, 판매 촉진 등을 위해 제품 및 기업에 관한 자료를 소비자들에게 직접 호소하여 제품의 우위성을 인식시키는 전시공간은?
    
    Top 5 most similar sentences in corpus:
    참가자에게 일정한 역할을 주어 실제적으로 연기를 시켜봄으로써 자기의 역할을 보다 확실히 인식할 수 있도록 체험학습을 시키는 교육방법은? (Score: 0.3399)
    검증(Validation) 검사 기법 중 개발자의 장소에서 사용자가 개발자 앞에서 행해지며, 오류와 사용상의 문제점을 사용자와 개발자가 함께 확인하면서 검사하는 기법은? (Score: 0.3320)
    물체의 표면에 침투력이 강한 적색 또는 형광성의 침투액을 표면 개구 결함에 침투시켜 직접 또는 자외선 등으로 관찰하여 결함장소와 크기를 판별하는 비파괴시험은? (Score: 0.3310)
    시각적 표시장치보다 청각적 표시장치를 사용하는 것이 더 유리한 경우는? (Score: 0.3293)
    지구단위계획구역의 지정목적을 이루기 위하여 지구단위계획에 포함될 수 있는 내용이 아닌 것은? (Score: 0.3268)


​    
    ======================


​    
    Query: 공조부하 중 현열과 잠열이 동시에 발생하는 것은?
    
    Top 5 most similar sentences in corpus:
    다음 중 냉방부하 계산 시 현열과 잠열 모두 고려하여야 하는 요소는? (Score: 0.5376)
    트랜잭션의 주요 특성 중 하나로 둘 이상의 트랜잭션이 동시에 병행 실행되는 경우 어느 하나의 트랜잭션 실행 중에 다른 트랜잭션의 연산이 끼어들 수 없음을 의미하는 것은? (Score: 0.4447)
    다음과 같은 조건에 있는 실의 틈새바람에 의한 현열부하는? (Score: 0.4239)
    두 가지 상태 중 하나가 고장 또는 결함으로 나타나는 비정상적인 사건은? (Score: 0.4096)
    다음과 같은 조건에 있는 실의 틈새바람에 의한 현열 부하량은? (Score: 0.3920)


​    
    ======================


​    
    Query: 건축물의 피난·방화구조 등의 기준에 관한 규칙상 방화구획의 설치기준 중 스프링클러를 설치한 10층 이하의 층은 바닥면적 몇 m2 이내마다 방화구획을 구획하여야 하는가?
    
    Top 5 most similar sentences in corpus:
    화재예방, 소방시설 설치·유지 및 안전관리에 관한 법령상 지하가는 연면적이 최소 몇 m2이상이어야 스프링클러설비를 설치하여야 하는 특정소방대상물에 해당하는가? (단, 터널은 제외한다.) (Score: 0.6573)
    피난기구의 화재안전기준에 따라 숙박시설·노유자시설 및 의료시설로 사용되는 층에 있어서는 그 층의 바닥면적이 몇 m2 마다 피난기구를 1개 이상 설치해야하는가? (Score: 0.6235)
    화재예방, 소방시설 설치·유지 및 안전관리에 관한 법령상 스프링클러설비를 설치하여야 하는 특정소방대상물의 기준으로 틀린 것은? (단, 위험물 저장 및 처리 시설 중 가스시설 또는 지하구는 제외한다.) (Score: 0.5706)
    스프링클러설비의 화재안전기준에 따른 특정소방대상물의 방호구역 층마다 설치하는 폐쇄형 스프링클러설비 유수검지장치의 설치 높이 기준은? (Score: 0.5572)
    화재예방, 소방시설 설치·유지 및 안전관리에 관한 법령상 건축허가등의 동의대상물의 범위 기준 중 틀린 것은? (Score: 0.5552)