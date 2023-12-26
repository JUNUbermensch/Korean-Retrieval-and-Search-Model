'''
여러 모델을 구현하여 직접 쿼리를 입력받고 해당 쿼리에 대한 검색능력을 비교함
동일 모델을 중복으로 사용하여 첫번째에는 document에서 해당 문서를 서치하는 데
두번째에는 해당 문서에서 query에 해당하는 내용을 추출하는 데 사용

1. context embedding을 document의 개별 문서 embedding 간의 코사인 유사도를 통해 비교하여
가장 유사도가 높은 개별문서를 추출함
2. 개별문서의 문장들을 list로 split한 다음 list 요소와 query를 vectorize하여서 다시 embedding 간의 코사인 유사도를 비교한 다음
가장 유사도가 높은 문장을 추출함 (따로 Reader를 만들지 않음)
모델은 1에서 썼던 모델을 2에서도 똑같이 사용함
예를 들어, BM25 모델을 개별 문장을 추출하는 데 사용했다면,
개별 문장을 split 한 다음 query와 리스트 내의 문장을 vectorize하는 데에도 BM25를 사용함
유사도를 비교할 때에는 가장 유사도 알고리즘 중 가장 성능이 좋았던 코사인 유사도를 사용함

- TF-IDF (Term Frequency-Inverse Document Frequency)
문서 내에서 단어의 빈도와 그 단어가 다른 문서들에 걸쳐 얼마나 희귀한지를 고려하여 가중치를 계산함
고전적인 방법론으로서 텍스트 검색에서 널리 사용되며 간단하고 효과적인 방법이지만 문맥이나 단어의 의미를 파악하지는 못함

- BM25 (Best Matching 25)
TF-IDF의 확장으로 검색 쿼리와 문서 간의 관련성을 평가하는 데 사용
정보 검색에서 매우 효과적이며 TF-IDF보다 더 진보된 기술로 여겨집니다. 문맥을 고려하지는 않지만 문서의 길이를 고려하여 보다 정교한 점수 지표를 제공함

- Inverted Index
단어와 그 단어가 포함된 문서의 목록을 매핑하는 데이터 구조
대규모 문서 집합에서 빠른 검색을 가능하게 하며 검색 엔진의 핵심 구성 요소 단 자체적으로는 문맥이나 의미를 분석하지 않음

- RNN (Recurrent Neural Network)
시퀀스 데이터 처리에 적합한 신경망으로 이전의 출력이 다음의 입력으로 활용됨
자연어 처리에서 중요한 역할을 하며 문맥을 고려할 수 있지만 긴 시퀀스에서는 효율성이 떨어질 수 있다는 단점이 존재
본 프로젝트에서 유일하게 학습 레이어를 쌓고 학습을 시킨 모델

- DPR (Dense Passage Retrieval)
질문과 문서를 밀집 벡터로 변환하여 매칭하여 비슷한 순으로 내림차순 정렬 top k 만큼의 값을 뽑아냄
질문에 가장 잘 맞는 문서를 찾는 데 효과적이며 특히 자연어 질의에 대한 정확한 답변 찾기에 적합함
긴 문서를 읽고 이해하는 데 적합함

- Haystack
retrieval augmented generation을 위한 end to end library로서 Transformer와 vector search를 기반으로 함
SQLite를 통해 document를 저장하고 TFIDFRetriever를 통해 document를 vectorize
query 역시 같은 retriever를 통해 vectorize하고 query와 유사도가 높은 document를 내림차순으로 정렬하여 top_k만큼 추출함
강력한 end to end 모델을 몇번의 라이브러리 call만으로 쉽게 구현할 수 있는 장점

- Transformers
Attention 메커니즘을 사용하여 입력 데이터의 모든 부분 간의 관계를 고려함
자연어 이해 및 생성 작업에서 뛰어난 성능을 보이며 복잡한 문맥과 의미를 파악하는 데 매우 효과적
본 프로젝트에서는 document가 한국어 이므로 kcbert의 개정판인 kcelectra 모델을 base 모델로 사용함

- Sentence Transformers
문장 또는 문단 전체의 의미를 포착하여 벡터로 변환함
문장 간의 의미적 유사성을 파악하는 데 사용되며 문맥을 포함한 텍스트의 의미를 잘 이해할 수 있음

- KoSimCSE-roberta (Roberta based - BERT based), paraphrase mpnet (based on MPNet), MSMARCO DistilBERT (BERT based)
최신 문장 임베딩 모델로 각기 다른 BERT based 모델을 사용
한국어 텍스트의 의미적 유사성을 판단하는 데 매우 효과적이며 특히 한국어 자연어 이해에 적합함
본 프로젝트에서 SOTA 모델의 예시로 가져옴
'''

import chardet
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import json
import numpy as np
import jpype
import torch.nn.functional as F
import gc
import os
import re
import time
from collections import defaultdict
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from haystack.nodes import TfidfRetriever
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from konlpy import init_jvm
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from haystack.document_stores.sql import SQLDocumentStore


#clear gpu memory cache

gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2'
model = None
jvmpath = jpype.getDefaultJVMPath()
okt = Okt()
# txt 파일이 너무 커서 heap size 늘림
init_jvm(jvmpath=jvmpath, max_heap_size=65536)
# 한국어 불용어 리스트 (nltk에서 korean stopwords 지원을 안해서 직접 적음)
stop_words = set([
    "가", "가까스로", "가령", "각", "각각", "각자", "각종", "갖고말하자면",
    "같다", "같이", "개의치않고", "거니와", "거바", "거의", "것", "것과 같이", "것들",
    "게다가", "게우다", "겨우", "견지에서", "결과에 이르다", "결국", "결론을 낼 수 있다", "겸사겸사",
    "고려하면", "고로", "곧", "공동으로", "과", "과연", "관계가 있다", "관계없이", "관련이 있다",
    "관하여", "관한", "관해서는", "구", "구체적으로", "구토하다", "그", "그들", "그때", "그래",
    "그래도", "그래서", "그러나", "그러니", "그러니까", "그러면", "그러므로", "그러한즉", "그런 까닭에",
    "그런데", "그런즉", "그럼", "그럼에도 불구하고", "그렇게 함으로써", "그렇지", "그렇지 않다면",
    "그렇지 않으면", "그렇지만", "그렇지않으면", "그리고", "그리하여", "그만이다", "그에 따르는",
    "그위에", "그저", "그중에서", "그치지 않다", "근거로", "근거하여", "기대여", "기점으로", "기준으로",
    "기타", "까닭으로", "까악", "까지", "까지 미치다", "까지도", "꽈당", "끙끙", "끼익", "나", "나머지는",
    "남들", "남짓", "너", "너희", "너희들", "네", "넷", "년", "논하지 않다", "놀라다", "누가 알겠는가",
    "누구", "다른", "다른 방면으로", "다만", "다섯", "다소", "다수", "다시 말하자면", "다시말하면", "다음",
    "다음에", "다음으로", "단지", "답다", "당신", "당장", "대로 하다", "대하면", "대하여", "대해 말하자면",
    "대해서", "댕그", "더구나", "더군다나", "더라도", "더불어", "더욱더", "더욱이는", "도달하다", "도착하다",
    "동시에", "동안", "된바에야", "된이상", "두번째로", "둘", "둥둥", "뒤따라", "뒤이어", "든간에", "들",
    "등", "등등", "딩동", "따라", "따라서", "따위", "따지지 않다", "딱", "때", "때가 되어", "때문에", "또",
    "또한", "뚝뚝", "라 해도", "령", "로", "로 인하여", "로부터", "로써", "륙", "를", "마음대로", "마저", "마저도",
    "마치", "막론하고", "만 못하다", "만약", "만약에", "만은 아니다", "만이 아니다", "만일", "만큼", "말하자면",
    "말할것도 없고", "매", "매번", "메쓰겁다", "몇", "모", "모두", "무렵", "무릎쓰고", "무슨", "무엇", "무엇때문에",
    "물론", "및", "바꾸어말하면", "바꾸어말하자면", "바꾸어서 말하면", "바꾸어서 한다면", "바꿔 말하면", "바로", "바와같이",
    "밖에 안된다", "반대로", "반대로 말하자면", "반드시", "버금", "보는데서", "보다더", "보드득", "본대로", "봐", "봐라",
    "부류의 사람들", "부터", "불구하고", "불문하고", "붕붕", "비걱거리다", "비교적", "비길수 없다", "비로소", "비록",
    "비슷하다", "비추어 보아", "비하면", "뿐만 아니라", "뿐만아니라", "뿐이다", "삐걱", "삐걱거리다", "사", "삼", "상대적으로 말하자면",
    "생각한대로", "설령", "설마", "설사", "셋", "소생", "소인", "솨", "쉿", "습니까", "습니다", "시각", "시간", "시작하여",
    "시초에", "시키다", "실로", "심지어", "아", "아니", "아니나다를가", "아니라면", "아니면", "아니었다면", "아래윗", "아무거나",
    "아무도", "아야", "아울러", "아이", "아이고", "아이구", "아이야", "아이쿠", "아하", "아홉", "안 그러면", "않기 위하여",
    "않기 위해서", "알 수 있다", "알았어", "앗", "앞에서", "앞의것", "야", "약간", "양자", "어", "어기여차", "어느",
    "어느 년도", "어느것", "어느곳", "어느때", "어느쪽", "어느해", "어디", "어때", "어떠한", "어떤", "어떤것", "어떤것들",
    "어떻게", "어떻해", "어이", "어째서", "어쨋든", "어쩌라고", "어쩌면", "어쩌면 해도", "어쩌다", "어쩔수 없다", "어찌",
    "어찌됏든", "어찌됏어", "어찌하든지", "어찌하여", "언제", "언젠가", "얼마", "얼마 안 되는 것", "얼마간", "얼마나", "얼마든지",
    "얼마만큼", "얼마큼", "엉엉", "에", "에 가서", "에 달려 있다", "에 대해", "에 있다", "에 한하다", "에게", "에서", "여", "여기",
    "여덟", "여러분", "여보시오", "여부", "여섯", "여전히", "여차", "연관되다", "연이서", "영", "영차", "옆사람", "예", "예를 들면",
    "예를 들자면", "예컨대", "예하면", "오", "오로지", "오르다", "오자마자", "오직", "오호", "오히려", "와", "와 같은 사람들",
    "와르르", "와아", "왜", "왜냐하면", "외에도", "요만큼", "요만한 것", "요만한걸", "요컨대", "우르르", "우리", "우리들", "우선", "우에 종합한것과같이",
    "운운", "월", "위에서 서술한바와같이", "위하여", "위해서", "윙윙", "육", "으로", "으로 인하여", "으로서", "으로써", "을", "응", "응당",
    "의", "의거하여", "의지하여", "의해", "의해되다", "의해서", "이", "이 되다", "이 때문에", "이 밖에", "이 외에", "이 정도의", "이것",
    "이곳", "이때", "이라면", "이래", "이러이러하다", "이러한", "이런", "이럴정도로", "이렇게 많은 것", "이렇게되면", "이렇게말하면",
    "이렇구나", "이로 인하여", "이르기까지", "이리하여", "이만큼", "이번", "이봐", "이상", "이어서", "이었다", "이와 같다", "이와 같은",
    "이와 반대로", "이와같다면", "이외에도", "이용하여", "이유만으로", "이젠", "이지만", "이쪽", "이천구", "이천육", "이천칠", "이천팔",
    "인 듯하다", "인젠", "일", "일것이다", "일곱", "일단", "일때", "일반적으로", "일지라도", "임에 틀림없다", "입각하여", "입장에서", "잇따라", "있다",
    "자", "자기", "자기집", "자마자", "자신", "잠깐", "잠시", "저", "저것", "저것만큼", "저기", "저쪽", "저희", "전부",
    "전자", "전후", "점에서 보아", "정도에 이르다", "제", "제각기", "제외하고",
    "조금", "조차", "조차도", "졸졸", "좀", "좋아", "좍좍", "주룩주룩", "주저하지 않고",
    "줄은 몰랏다", "줄은모른다", "중에서", "중의하나", "즈음하여", "즉", "즉시", "지든지",
    "지만", "지말고", "진짜로", "쪽으로", "차라리", "참", "참나", "첫번째로", "쳇", "총적으로",
    "총적으로 말하면", "총적으로 보면", "칠", "콸콸", "쾅쾅", "쿵", "타다", "타인", "탕탕",
    "토하다", "통하여", "툭", "퉤", "틈타", "팍", "팔", "퍽", "펄렁", "하", "하게될것이다",
    "하게하다", "하겠는가", "하고 있다", "하고있었다", "하곤하였다", "하구나", "하기 때문에",
    "하기 위하여", "하기는한데", "하기만 하면", "하기보다는", "하기에", "하나", "하느니",
    "하는 김에", "하는 편이 낫다", "하는것도", "하는것만 못하다", "하는것이 낫다", "하는바",
    "하더라도", "하도다", "하도록시키다", "하도록하다", "하든지", "하려고하다", "하마터면",
    "하면 할수록", "하면된다", "하면서", "하물며", "하여금", "하여야", "하자마자", "하지 않는다면",
    "하지 않도록", "하지마", "하지마라", "하지만", "하하", "한 까닭에", "한 이유는", "한 후",
    "한다면", "한다면 몰라도", "한데", "한마디", "한적이있다", "한켠으로는", "한항목", "할 따름이다",
    "할 생각이다", "할 줄 안다", "할 지경이다", "할 힘이 있다", "할때", "할만하다", "할망정", 
    "할뿐", "할수있다", "할수있어", "할줄알다", "할지라도", "할지언정", "함께", "해도된다", "해도좋다",
    "해봐요", "해서는 안된다", "해야한다", "해요", "했어요", "향하다", "향하여", "향해서", "허", 
    "허걱", "허허", "헉", "헉헉", "헐떡헐떡", "형식으로 쓰여", "혹시", "혹은", "혼자", "훨씬", 
    "휘익", "휴", "흐흐", "흥", "힘입어"])

# txt 파일을 utf-8 형식으로 변환
def converter(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    encode = chardet.detect(rawdata)

    with open(file_path, 'r', encoding=encode['encoding']) as file:
        content = file.read()
        
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# 2차 텍스트 정제 함수
# 추출한 문서를 정제하는 데 사용되며 문장 하나가 리스트의 요소 하나의 역할을 함
# stop words들을 걸러냄
def deep_preprocess_text(text):  
    text = BeautifulSoup(text, 'html.parser').get_text(separator='', strip=True)
    tokens = text.split('\n')
    tokens = [token for token in tokens if stop_words not in okt.morphs(token)]
    tokens = tokens[0].split('.')

    return tokens

# 1차 텍스트 정제 함수 
# HTML tags, scripts, styles, and extra whitespace들을 쳐낸 후 개행을 기준으로 텍스트를 요소 별로 분리하여 리스트로 추출
def preprocess_text(text): 
    text = re.sub(r'\[\[파일:[^\]]+\]\]', '', text)
    text = re.sub(r'\{\{[^\}]+\}\}', '', text)
    text = re.sub(r'<math>[^<]+</math>', '', text)    
    text = BeautifulSoup(text, 'html.parser').get_text(separator='', strip=True)
    tokens = text.split('\n')
    tokens = [token for token in tokens if stop_words not in okt.morphs(token)]

    return tokens

# txt 파일을 읽기 위한 함수
def read_korean_wikipedia_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    processed_text = preprocess_text(text)
    
    return processed_text

# 유사도에는 코사인 유사도 함수를 사용
# 임베딩 간의 유사도를 측정하기에 가장 적합함
def jaccard_similarity(doc1, doc2):
    tokens1 = okt.nouns(doc1)
    tokens2 = okt.nouns(doc2)

    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 0

    jaccard_similarity = len(intersection) / len(union)

    return jaccard_similarity

# 샘플로 주어진 txt 파일의 특성을 이용해 context(topic)가 '''로 묶여 있다는 사실을 이용하여 정답 document를 추출
def golden_document(documents, query):
    query_with_quotes = f"'''{query}'''"

    for document in documents:
        # 빈 element 일 경우 skip
        if not document.strip():
            continue
        # '''로 묶인 context (여기서는 query 변수로 받음)가 나올 경우 해당 document를 추출하고 loop 문을 빠져나옴
        occurrence = document.count(query_with_quotes)

        if occurrence > 0:
            golden_document = document
            break

    return golden_document

# 추후 사용될 RNN 함수의 레이어를 받기위해 forward 함수의 구조를 정의함
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, batch_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 긴 document 이므로 LSTM 모델을 사용
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.batch_size = batch_size

    def forward(self, x):  
        try:
            embedded = self.embedding(x)
            rnn_out, _ = self.rnn(embedded)
            # overfitting 방지용 dropout을 사용
            rnn_out = self.dropout(rnn_out)
            # output에서 last hidden state만을 받아서 return함 중간 과정은 필요없으므로 [batch_size, sequence_length, hidden_size]
            output = self.fc(rnn_out[:, -1, :])
            
            return output
        except RuntimeError as e:
            return torch.tensor(x)

class SearchModel:
    def __init__(self, documents):
        # 초기화
        self.documents = documents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Haystack
        haystack_documents = [{"content": text} for text in documents]
        # SQLite를 사용해 document를 저장
        document_store = SQLDocumentStore(url="sqlite:///:memory:")
        document_store.write_documents(haystack_documents)
        # TfidfRetriever를 통해서 retrieve (query와의 유사도 계산할 수 있도록)
        self.retriever = TfidfRetriever(document_store=document_store)
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
        self.TFIDF_tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        # BM25
        # token을 count하여 vectorized matrix로 변환할 수 있도록 초기화
        self.count_vectorizer = CountVectorizer()
        # document의 토큰들을 vectorize
        self.term_freq_matrix = self.count_vectorizer.fit_transform(documents)
        # TFIDFTransformer를 통해 유사도를 계산할 수 있게 초기화
        self.tfidf_transformer = TfidfTransformer()
        # TFIDFTransformer에 fit하여서 유사도를 계산할 수 있도록 함
        self.bm25_tfidf_matrix = self.tfidf_transformer.fit_transform(self.term_freq_matrix)

        # DPR
        self.dpr_context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        self.dpr_question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
        self.dpr_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")

        # Inverted Index
        self.inverted_index = defaultdict(list)
        self._build_inverted_index()

        # RNN
        self.rnn_tokenizer = Tokenizer()
        self.rnn_tokenizer.fit_on_texts(self.documents)
        self.rnn_sequences = self.rnn_tokenizer.texts_to_sequences(self.documents)
        self.rnn_padded_sequences = pad_sequences(self.rnn_sequences)
        # RNN model finetuning
        self.rnn_model = self._build_rnn_model()
        
        # Transformers
        self.transformers_tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
        self.transformers_model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base", num_hidden_layers=6,
                                                                       num_attention_heads=8)
        # Sentence Transformer Model
        # paraphrase-MiniLM-L6-v2dms: 6레이어의 작은 구조를 가진 모델로 빠른 엑세스, 빠른 서치가 가능
        # 임베딩 간의 semantic similarity를 빠르게 계산할 수 있는 모델
        self.sentence_transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # SOTA Model
        self.KSCRoberta_model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
        self.KSCRoberta_tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
        self.msmarco_model  = SentenceTransformer('msmarco-distilbert-base-v2')
        self.paraphrase_mpnet_model = SentenceTransformer('paraphrase-mpnet-base-v2')

    def paraphrase_mpnet_search(self, query, top_k=1):
        query_embedding = self.paraphrase_mpnet_model.encode(query, convert_to_tensor=True)
        document_embeddings = self.paraphrase_mpnet_model.encode(self.documents, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        most_similar_indices = torch.argsort(cosine_scores, descending=True)[:top_k]

        return [self.documents[index] for index in most_similar_indices]
    
    def msmarco_search(self, query, top_k=1):
        query_embedding = self.msmarco_model.encode(query, convert_to_tensor=True)
        document_embeddings = self.msmarco_model.encode(self.documents, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        most_similar_indices = torch.argsort(cosine_scores, descending=True)[:top_k]

        return [self.documents[index] for index in most_similar_indices]
    
    def dpr_search(self, query, top_k=1):
        # search 함수에는 query와 document를 tokenize 한 다음 document의 element와 query 간의 코사인유사도가 가장 높은 값을 찾아내어 내림차순으로 sort하여 topk 만큼 추출함
        # 2개 빼고 모든 search 함수에는 이 방법을 사용함
        query_tokenized = self.dpr_tokenizer(query, return_tensors="pt")
        question_embedding = self.dpr_question_encoder(**query_tokenized)["pooler_output"]
        document_embeddings = self.dpr_context_encoder(input_ids=query_tokenized["input_ids"],
                                                       attention_mask=query_tokenized["attention_mask"])["pooler_output"]
        cosine_scores = util.pytorch_cos_sim(question_embedding, document_embeddings)[0]
        most_similar_indices = torch.argsort(cosine_scores, descending=True)[:top_k]

        return [self.documents[index] for index in most_similar_indices]
    
    def KSCRoberta_search(self, query, top_k=1):
        query_tokenized = self.KSCRoberta_tokenizer(query, padding=True, truncation=True, return_tensors='pt')
        query_embedding = self.KSCRoberta_model(**query_tokenized).last_hidden_state.mean(dim=1)
        document_embeddings = self.KSCRoberta_model(**self.KSCRoberta_tokenizer(self.documents, padding=True, truncation=True, return_tensors='pt')).last_hidden_state.mean(dim=1)

        cosine_scores = F.cosine_similarity(query_embedding, document_embeddings)
        most_similar_indices = torch.argsort(cosine_scores, descending=True)[:top_k]

        return [self.documents[index] for index in most_similar_indices]

    def sentence_transformers_search(self, query, top_k=1):
        query_embedding = self.sentence_transformer_model.encode(query, convert_to_tensor=True)
        document_embeddings = self.sentence_transformer_model.encode(self.documents, convert_to_tensor=True)

        cosine_scores = F.cosine_similarity(query_embedding, document_embeddings)
        most_similar_indices = torch.argsort(cosine_scores, descending=True)[:top_k]

        return [self.documents[index] for index in most_similar_indices]
        
    def _build_inverted_index(self):
        # documnet에서 index와 실제 document의 element를 루프하면서 추출
        # document의 element에서 단어를 추출하여 index를 집어넣음
        for doc_id, doc in enumerate(self.documents):
            terms = set(doc.split())
            for term in terms:
                self.inverted_index[term].append(doc_id)

    def inverted_index_search(self, query, top_k=1):
        # query 벡터를 0벡터로 초기화
        query_vector = np.zeros(len(self.inverted_index))
        feature_names = list(self.inverted_index.keys())
        
        # query에서 불용어를 제외하고 단어를 추출
        for term in query.split():
            if term in feature_names and term not in stop_words:
                term_index = feature_names.index(term)
                query_vector[term_index] = 1
        
        # 사전에 추출했던 document의 element와 query가 가장 유사한 케이스를 추출하여 그 정도와 함께 dict에 넣음
        matching_documents = defaultdict(float)
        for term_index, term in enumerate(feature_names):
            if query_vector[term_index] > 0:
                for doc_index in self.inverted_index[term]:
                    matching_documents[doc_index] += 1
        
        # dict를 유사도 순으로 sort함 dictionary이므로 sorting이 빠름
        sorted_documents = sorted(matching_documents.keys(), key=lambda x: matching_documents[x], reverse=True)

        return [self.documents[index] for index in sorted_documents[:top_k]]
    
    def _build_rnn_model(self, patience=3):
        # document의 수만큼 0, 1, 2, 3, 4, ...로 indexing
        self.document_labels = [i for i in range(len(self.documents))]  # Assign class labels to documents
        
        # 아까 build한 구조대로 layer를 정의하여 집어넣음
        model = RNNModel(vocab_size=len(self.rnn_tokenizer.word_index) + 1, embed_dim=256, hidden_size=256, num_classes=len(self.documents), batch_size=16)
        #  정보 이론적인 확률을 추출해야 하므로 크로스 엔트로피 로스를 사용
        criterion = nn.CrossEntropyLoss()
        # weight decay를 사용하여 overfitting을 방지
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        # 러닝레이트 스케줄러를 사용하여 loss값 수렴이 빨리 되도록 함
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

        x_train = torch.tensor(self.rnn_padded_sequences, dtype=torch.long)
        y_train = torch.tensor(self.document_labels, dtype=torch.long)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        try:
            x_train, x_val, y_train, y_val = train_test_split(
                self.rnn_padded_sequences, 
                self.document_labels, 
                test_size=0.2,
                train_size=0.8,
                random_state=42
            )
        except ValueError as e:
             x_train, x_val, y_train, y_val = [], [], [], []
        best_loss = float('inf')
        current_patience = 0

        for epoch in range(30):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            outputs = model(torch.tensor(x_train, dtype=torch.long).to(device))
            loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_train_loss = total_loss / len(x_train)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                val_outputs = model(torch.tensor(x_val, dtype=torch.long).to(device))
                val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.long).to(device)).item()
                total_val_loss += val_loss
                avg_val_loss = total_val_loss / len(x_val)

            print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                current_patience = 0
            else:
                current_patience += 1

            scheduler.step(avg_val_loss)
            if current_patience >= patience:
                print(f'Early stopping after {epoch + 1} epochs.')
                break

        return model
        
    def rnn_search(self, query, top_k=1):
        # query를 tokenize
        sequence = self.rnn_tokenizer.texts_to_sequences(query)
        # padding
        padded_sequence = pad_sequences(sequence, maxlen=len(self.rnn_padded_sequences[0]))
        x_input = torch.tensor(padded_sequence, dtype=torch.long).to(next(self.rnn_model.parameters()).device)

        output = self.rnn_model(x_input)
        # 확률에 softmax를 취하여 정규화
        probabilities = nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]
        # 가장 가능성이 높은 순서대로 sort한 다음 top_k 값만큼 추출
        most_probable_indices = probabilities.argsort()[-top_k:][::-1]
        
        return [self.documents[index] for index in most_probable_indices]


    def tfidf_search(self, query, top_k=1):
        query_vector = self.tfidf_vectorizer.transform([query])
        query_similarity = cosine_similarity(query_vector, self.TFIDF_tfidf_matrix)[0]
        most_similar_indices = query_similarity.argsort()[-top_k:][::-1]
        
        return [self.documents[index] for index in most_similar_indices]

    def bm25_search(self, query, top_k=1):
        query_vector = self.count_vectorizer.transform([query])
        query_tfidf = self.tfidf_transformer.transform(query_vector)
        bm25_similarity = cosine_similarity(query_tfidf, self.bm25_tfidf_matrix)[0]
        most_similar_indices = bm25_similarity.argsort()[-top_k:][::-1]
        
        return [self.documents[index] for index in most_similar_indices]

    def transformers_search(self, query, top_k=1):
        encoded_input = self.transformers_tokenizer(query, return_tensors='pt')
        logits = self.transformers_model(**encoded_input).logits
        probabilities = nn.functional.softmax(logits, dim=1).detach().numpy()[0]
        most_probable_indices = probabilities.argsort()[-top_k:][::-1]
        
        return [self.documents[index] for index in most_probable_indices]
    
    def haystack_search(self, query, top_k=1):
            retriever_results = self.retriever.retrieve(query=query, top_k=top_k)
            
            return retriever_results
            
if __name__ == "__main__":
    # file_path = '/data/workspace/junow/haystack/small_example.txt' # for test
    # 시간이 너무 오래걸려서 원본 파일의 일부분만을 추출한 small example txt 파일을 사용
    file_path = '/data/workspace/junow/haystack/kowiki-20220220-001.txt'
    converter(file_path)
    documents = read_korean_wikipedia_file(file_path)
    search_model = SearchModel(documents)
    context_list = []
    top_k = 1
    if input('enter 0 to move on or enter other word to evaludate models: ') != '0':
        for document in documents:
            # 검색 할 수 있는 단어들을 전부 list에 넣음 (빈 문자열은 제외)
            matches = re.findall(r"'''(.+?)'''", document)
            context_list.extend(matches)
            print(context_list)
        for n, (model_name, search_function) in enumerate([
            ("TF-IDF", search_model.tfidf_search), # 기본 모델
            ("BM25", search_model.bm25_search), # 기본 모델
            ("Inverted Index", search_model.inverted_index_search), # 기본 모델
            ("Haystack", search_model.haystack_search), # 기본 모델
            ("RNN", search_model.rnn_search), # 기본모델, 학습 o
            ("DPR", search_model.dpr_search), # finetuned model, 학습 x
            ("Transformers", search_model.transformers_search), # finetuned model, 학습 x
            ("Sentence Transformers", search_model.sentence_transformers_search), # finetuned model, 학습 x
            ("KSCRoberta Search", search_model.KSCRoberta_search), # pretrained model, finetuned
            ("MSMARCO DistilBERT", search_model.msmarco_search), # pretrained model, finetuned
            ("Paraphrase-MPNet", search_model.paraphrase_mpnet_search), # pretrained model, finetuned
        ]):
            TFIDF_similarity = 0
            BM25_similarity = 0
            InvertedIndex_similarity = 0
            Haystack_similarity = 0
            RNN_similarity = 0
            DPR_similarity = 0
            Transformers_similarity = 0
            SentenceTransformers_similarity = 0
            KSCRoberta_similarity = 0
            MSMARCO_similarity = 0
            Paraphrase_similarity = 0
            
            for m, context in enumerate(context_list):
                try:
                    result = search_function(context, top_k=top_k)
                # context를 아예 찾지 못할 경우를 대비해서 try except문을 사용
                except TypeError as e:
                    print(e)
                    result = []
                    print('0%')
                    continue
                except ValueError as e:
                    print(e)
                    result = []
                    print('0%')
                    continue
                GoldenDocument = golden_document(documents, context)
                m += 1
                # 1번 TF-IDF
                if n == 1:
                    print(GoldenDocument, context)
                    TFIDF_similarity += jaccard_similarity(result[0], GoldenDocument)
                    TFIDF_similarity = TFIDF_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {TFIDF_similarity:.2%}\n")
                # 2번 BM25
                elif n == 2:
                    BM25_similarity += jaccard_similarity(result[0], GoldenDocument)
                    BM25_similarity = BM25_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {BM25_similarity:.2%}\n")
                # 3번 Inverted Index
                elif n == 3:
                    InvertedIndex_similarity += jaccard_similarity(result[0], GoldenDocument)
                    InvertedIndex_similarity = InvertedIndex_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {InvertedIndex_similarity:.2%}\n")
                # 4번 Haystack
                elif n == 4:
                    Haystack_similarity += jaccard_similarity(result[0], GoldenDocument)
                    Haystack_similarity = Haystack_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {Haystack_similarity:.2%}\n")
                # 5번 RNN
                elif n == 5:
                    RNN_similarity += jaccard_similarity(result[0], GoldenDocument)
                    RNN_similarity = RNN_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {RNN_similarity:.2%}\n")
                # 6번 DPR
                elif n == 6:
                    DPR_similarity += jaccard_similarity(result[0], GoldenDocument)
                    DPR_similarity = DPR_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {DPR_similarity:.2%}\n")
                # 7번 Transformers
                elif n == 7:
                    Transformers_similarity += jaccard_similarity(result[0], GoldenDocument)
                    Transformers_similarity = Transformers_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {Transformers_similarity:.2%}\n")
                # 8번 Sentence Transforemrs
                elif n == 8:
                    SentenceTransformers_similarity += jaccard_similarity(result[0], GoldenDocument)
                    SentenceTransformers_similarity = SentenceTransformers_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {SentenceTransformers_similarity:.2%}\n")
                # 9번 KSCRoberTa (SOTA, RoberTa 기반)
                elif n == 9:
                    KSCRoberta_similarity += jaccard_similarity(result[0], GoldenDocument)
                    KSCRoberta_similarity = KSCRoberta_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {KSCRoberta_similarity:.2%}\n")
                # 10번 msmarco (SOTA, BERT 기반)
                elif n == 10:
                    MSMARCO_similarity += jaccard_similarity(result[0], GoldenDocument)
                    MSMARCO_similarity = MSMARCO_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {MSMARCO_similarity:.2%}\n")
                # 11번 paraphrase mpnet (SOTA, mpnet 기반)
                else:
                    Paraphrase_similarity += jaccard_similarity(result[0], GoldenDocument)
                    Paraphrase_similarity = Paraphrase_similarity / m
                    print(f"\naverage {model_name}'s similarity to a label document: {Paraphrase_similarity:.2%}\n")        
    
        print(TFIDF_similarity)
        print(BM25_similarity)
        print(InvertedIndex_similarity)
        print(Haystack_similarity)
        print(RNN_similarity)
        print(DPR_similarity)
        print(Transformers_similarity)
        print(SentenceTransformers_similarity)
        print(KSCRoberta_similarity)
        print(MSMARCO_similarity)
        print(Paraphrase_similarity)
    
    else:
        while input('enter 0 to break or enter other word to move on: ') != '0':
            # 일단 query를 입력
            query = input("enter your search query: ")
            # query에 맞는 context (topic)을 입력
            contexts = input("enter your topic of search query: ")
            messages = [{"role": "context", "content": context} for context in contexts]
            messages.append({"role": "user", "content": query})
            # t5 search (reader)
            input_q = json.dumps({
            "model": "t5-small-fid",
            "mode": "", 
            "messages": messages})
            response = requests.post('http://211.39.140.48:9090/predictions/temp', data=input_q)
            t5_prediction_result = response.json()
            print("\nT5-large Prediction Result:", t5_prediction_result)
            # 모델 별로 결과값을 추출하게 함
            for n, (model_name, search_function) in enumerate([
                ("TF-IDF", search_model.tfidf_search), # 기본 모델
                ("BM25", search_model.bm25_search), # 기본 모델
                ("Inverted Index", search_model.inverted_index_search), # 기본 모델
                ("Haystack", search_model.haystack_search), # 기본 모델
                ("RNN", search_model.rnn_search), # 기본모델, 학습 o
                ("DPR", search_model.dpr_search), # finetuned model, 학습 x
                ("Transformers", search_model.transformers_search), # finetuned model, 학습 x
                ("Sentence Transformers", search_model.sentence_transformers_search), # finetuned model, 학습 x
                ("KSCRoberta Search", search_model.KSCRoberta_search), # pretrained model, finetuned
                ("MSMARCO DistilBERT", search_model.msmarco_search), # pretrained model, finetuned
                ("Paraphrase-MPNet", search_model.paraphrase_mpnet_search), # pretrained model, finetuned
            ]):
                try:
                    result = search_function(contexts, top_k=top_k)
                # context를 아예 찾지 못할 경우를 대비해서 try except문을 사용
                except TypeError as e:
                    print(e)
                    result = []
                    print('0%')
                    continue
                except ValueError as e:
                    print(e)
                    result = []
                    print('0%')
                    continue
                # 결과물 추출 (여기에서의 결과물이란 document에서 해당하는 context에 관한 문서임)
                print(f"\n{model_name} Search in Document:", result)
                print('')
                try:
                    # 아까 설정했던 정답 문서 추출하는 함수
                    # wekipedia라고 모든 topic이 있진 않으므로 검색에 실패할 경우를 대비해 try except
                    GoldenDocument = golden_document(documents, contexts)
                    similarity = jaccard_similarity(result[0], GoldenDocument)
                    print(f"\n{model_name}'s similarity to a label document: {similarity:.2%}\n")
                except Exception as e:
                    similarity = 1
                    print(e)
                    print("can't find the topic in the document. please write the right word for searching in the document")
                # 추출된 result document element 그러니까 context에 해당하는 문서에서 query에 해당하는 부분을 다시 serach하여 출력하는 부분임
                if similarity * 10 > 1:
                    # document에서 context 단어에 해당하는 문서를 찾고 그 문서에서 다시 query에 해당하는 part를 찾음
                    # 처음에는 GPT랑 랭체인을 통해 QA를 하려 했으나 잘 안되서 search model을 한번 더 써보기로 함
                    # search 모델을 reader 모델로 쓰는 것이 가장 효과가 좋았음
                    # 변수를 쓰려고했더니 파이썬에서 이를 class의 함수로 인식을 해버려서 그냥 새로 deep search model을 만듦
                    # 구조는 같음
                    try:
                        # enumerate를 통해 인덱스를 추출한다음 같은 모델을 중복하여 사용함
                        # 1번: TF-IDF
                        if n == 1:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.tfidf_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        # 2번 BM25
                        elif n == 2:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.bm25_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        # 3번 Inverted Index
                        elif n == 3:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.inverted_index_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        # 4번 Haystack
                        elif n == 4:
                            candidate_list = deep_preprocess_text(result['content'])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.haystack_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                    # 5번 RNN
                        elif n == 5:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.rnn_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        # 6번 DPR
                        elif n == 6:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.dpr_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        # 7번 Transformers
                        elif n == 7:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.transformers_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        # 8번 Sentence Transforemrs
                        elif n == 8:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.sentence_transformers_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        # 9번 KSCRoberTa (SOTA, RoberTa 기반)
                        elif n == 9:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.KSCRoberta_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        # 10번 msmarco (SOTA, BERT 기반)
                        elif n == 10:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.msmarco_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        # 11번 paraphrase mpnet (SOTA, mpnet 기반)
                        else:
                            candidate_list = deep_preprocess_text(result[0])
                            deep_search_model = SearchModel(candidate_list)
                            ans = deep_search_model.paraphrase_mpnet_search(query, top_k=top_k)
                            print(f"\n{model_name}'s answer in Document:", ans)
                        time.sleep(100)
                    # 결과물이 너무 빨리 넘어가서 time.sleep 메서드를 사용함
                    # 마찬가지로 결과물을 못찾을 경우를 대비하여 try except 문을 사용
                    except TypeError as e:
                        print(e)
                        print("[]")
                        time.sleep(100)
                        continue
                    except ValueError as e:
                        print(e)
                        print("[]")
                        time.sleep(100)
                        continue
                    except RuntimeError as e:
                        print(e)
                        print("[]")
                        time.sleep(100)
                        continue
                else:
                    time.sleep(100)
                    print("[]")
                    continue


'''
- 결론
document 추출하는 로직은 잘 짜여진 것 같다
Reader를 정석으로 구현하지는 못했지만 Retrival 모델을 
Reader처럼 사용하는 로직을 통해 Reader의 역할을 수행하는 부분을 완성해내는 데 성공했다
임시 해결책처럼 보이기는 하나 GPT2 모델을 사용한 reader 보다는 효과적인 결과가 나오는 것으로 판단해서
이 방법을 계속 사용하기로 했다
'''

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
# export LD_LIBRARY_PATH=/path/to/tensorrt/lib:$LD_LIBRARY_PATH
# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python