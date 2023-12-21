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
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from konlpy import init_jvm
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

gc.collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['CUDA_VISIBLE_DEVICES']='2, 3'
torch.cuda.empty_cache()
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

def deep_preprocess_text(text):
    text = re.sub(r'\[\[파일:[^\]]+\]\]', '', text)
    text = re.sub(r'\{\{[^\}]+\}\}', '', text)
    text = re.sub(r'<math>[^<]+</math>', '', text)  
    text.replace('[', '')
    text.replace(']', '')    
    text = BeautifulSoup(text, 'html.parser').get_text(separator='', strip=True)
    tokens = text.split('\n')
    tokens = [token for token in tokens if stop_words not in okt.morphs(token)]
    tokens = tokens[0].split('.')

    return tokens

def preprocess_text(text): 
    text = re.sub(r'\[\[파일:[^\]]+\]\]', '', text)
    text = re.sub(r'\{\{[^\}]+\}\}', '', text)
    text = re.sub(r'<math>[^<]+</math>', '', text)  
    text.replace('[', '')
    text.replace(']', '')    
    text = BeautifulSoup(text, 'html.parser').get_text(separator='', strip=True)
    tokens = text.split('\n')
    tokens = [token for token in tokens if stop_words not in okt.morphs(token)]

    return tokens

def read_korean_wikipedia_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    processed_text = preprocess_text(text)
    
    return processed_text

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

def golden_document(documents, query):
    query_with_quotes = f"'''{query}'''"

    for document in documents:
        if not document.strip():
            continue
        occurrences = document.count(query_with_quotes)

        if occurrences > 0:
            golden_document = document
            break

    return golden_document

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, batch_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.batch_size = batch_size

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out[:, -1, :])
        
        return output

class SearchModel:
    def __init__(self, documents):
        self.documents = documents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
        self.TFIDF_tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        # BM25
        self.count_vectorizer = CountVectorizer()
        self.term_freq_matrix = self.count_vectorizer.fit_transform(documents)
        self.tfidf_transformer = TfidfTransformer()
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
        self.sentence_transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # SOTA Model
        self.KSCRoberta_model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
        self.KSCRoberta_tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')

    def dpr_search(self, query, top_k=1):
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
        for doc_id, doc in enumerate(self.documents):
            terms = set(doc.split())
            for term in terms:
                self.inverted_index[term].append(doc_id)

    def _build_rnn_model(self, patience=3):
        self.document_labels = [i for i in range(len(self.documents))]  # Assign class labels to documents
        
        model = RNNModel(vocab_size=len(self.rnn_tokenizer.word_index) + 1, embed_dim=256, hidden_size=256, num_classes=len(self.documents), batch_size=16)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

        x_train = torch.tensor(self.rnn_padded_sequences, dtype=torch.long)
        y_train = torch.tensor(self.document_labels, dtype=torch.long)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        x_train, x_val, y_train, y_val = train_test_split(
            self.rnn_padded_sequences, 
            self.document_labels, 
            test_size=0.2,
            train_size=0.8,
            random_state=42
        )
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
        sequence = self.rnn_tokenizer.texts_to_sequences(query)
        padded_sequence = pad_sequences(sequence, maxlen=len(self.rnn_padded_sequences[0]))
        x_input = torch.tensor(padded_sequence, dtype=torch.long).to(next(self.rnn_model.parameters()).device)

        output = self.rnn_model(x_input)
        probabilities = nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]
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

    def inverted_index_search(self, query, top_k=1):
        query_vector = np.zeros(len(self.inverted_index))
        feature_names = list(self.inverted_index.keys())
        
        for term in query.split():
            if term in feature_names and term not in stop_words:
                term_index = feature_names.index(term)
                query_vector[term_index] = 1
        
        matching_documents = defaultdict(float)
        for term_index, term in enumerate(feature_names):
            if query_vector[term_index] > 0:
                for doc_index in self.inverted_index[term]:
                    matching_documents[doc_index] += 1

        sorted_documents = sorted(matching_documents.keys(), key=lambda x: matching_documents[x], reverse=True)

        return [self.documents[index] for index in sorted_documents[:top_k]]

    def transformers_search(self, query, top_k=1):
        encoded_input = self.transformers_tokenizer(query, return_tensors='pt')
        logits = self.transformers_model(**encoded_input).logits
        probabilities = nn.functional.softmax(logits, dim=1).detach().numpy()[0]
        most_probable_indices = probabilities.argsort()[-top_k:][::-1]
        
        return [self.documents[index] for index in most_probable_indices]
            
if __name__ == "__main__":
    file_path = ''
    documents = read_korean_wikipedia_file(file_path)
    search_model = SearchModel(documents)
    
    while input('enter 0 to break or enter other word to move on: ') != '0':
        query = input("enter your search query: ")
        contexts = input("enter your topic of search query: ")
        messages = [{"role": "context", "content": context} for context in contexts]
        messages.append({"role": "user", "content": query})
       
        input_q = json.dumps({
        "model": "t5-small-fid",
        "mode": "", 
        "messages": messages})
        response = requests.post('', data=input_q)
        t5_prediction_result = response.json()
        print("\nT5-large Prediction Result:", t5_prediction_result)
        for n, (model_name, search_function) in enumerate([
            ("TF-IDF", search_model.tfidf_search), # 기본 모델
            ("BM25", search_model.bm25_search), # 기본 모델
            ("Inverted Index", search_model.inverted_index_search), # 기본 모델
            ("RNN", search_model.rnn_search), # 기본모델, 학습 o
            ("DPR", search_model.dpr_search), # finetuned model, 학습 x
            ("Transformers", search_model.transformers_search), # finetuned model, 학습 x
            ("Sentence Transformers", search_model.sentence_transformers_search), # finetuned model, 학습 x
            ("KSCRoberta Search", search_model.KSCRoberta_search), # pretrained model, finetuned
        ]):          
            top_k = 1
            try:
                result = search_function(contexts, top_k=top_k)
            except TypeError as e:
                print(e)
                print("[]")
                print('0%')
                continue
            except ValueError as e:
                print(e)
                print("[]")
                print('0%')
                continue
            result[0] = result[0].replace('[','').replace(']','')
            print(f"\n{model_name} Search in Document:", result)
            print('')
            try:
                GoldenDocument = golden_document(documents, contexts)
                similarity = jaccard_similarity(result[0], GoldenDocument)
                print(f"\nsimilarity to a label document: {similarity:.2%}\n")
            except Exception as e:
                print(e)
                print("can't find the topic in the document. please write the right word for searching in the document")
            candidate_list = deep_preprocess_text(result[0])
            deep_search_model = SearchModel(candidate_list)
            try:
                if n == 1:
                    ans = deep_search_model.tfidf_search(query, top_k=top_k)
                    print(f"\n{model_name}'s answer in Document:", ans)
                    print('')
                elif n == 2:
                    ans = deep_search_model.bm25_search(query, top_k=top_k)
                    print(f"\n{model_name}'s answer in Document:", ans)
                    print('')
                elif n == 3:
                    ans = deep_search_model.inverted_index_search(query, top_k=top_k)
                    print(f"\n{model_name}'s answer in Document:", ans)
                    print('')
                elif n == 4:
                    ans = deep_search_model.rnn_search(query, top_k=top_k)
                    print(f"\n{model_name}'s answer in Document:", ans)
                    print('')
                elif n == 5:
                    ans = deep_search_model.dpr_search(query, top_k=top_k)
                    print(f"\n{model_name}'s answer in Document:", ans)
                    print('')
                elif n == 6:
                    ans = deep_search_model.transformers_search(query, top_k=top_k)
                    print(f"\n{model_name}'s answer in Document:", ans)
                    print('')
                elif n == 7:
                    ans = deep_search_model.sentence_transformers_search(query, top_k=top_k)
                    print(f"\n{model_name}'s answer in Document:", ans)
                    print('')
                else:
                    ans = deep_search_model.KSCRoberta_search(query, top_k=top_k)
                    print(f"\n{model_name}'s answer in Document:", ans)
                    print('')
                time.sleep(10)
            except TypeError as e:
                print(e)
                print("[]")
                continue
            except ValueError as e:
                print(e)
                print("[]")
                print('0%')
                continue

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/lib
# export LD_LIBRARY_PATH=/path/to/tensorrt/lib:$LD_LIBRARY_PATH