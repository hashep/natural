import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# 불용어 및 스테머 준비
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# 텍스트 전처리 함수
def preprocess_text(text):
    text = str(text).lower()

    # 영어와 공백만 남기기
    text = re.sub(r'[^a-z\s]', '', text)

    # 단어 분리
    tokens = word_tokenize(text)

    # 불용어 제거 + 어간 추출
    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    return ' '.join(tokens)

# 데이터셋 불러오기
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=['label', 'message']
)

# ham=0, spam=1
df['label_num'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# 전처리
df['message_clean'] = df['message'].apply(preprocess_text)

# 학습용 / 테스트용 분리
X_train, X_test, y_train, y_test = train_test_split(
    df['message_clean'],
    df['label_num'],
    test_size=0.2,
    random_state=42
)

# TF-IDF 변환
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 모델 학습
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 예측
y_pred = model.predict(X_test_tfidf)

# 성능 출력
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# 사용자 입력 예측
def predict_spam(text):
    text_clean = preprocess_text(text)
    text_tfidf = vectorizer.transform([text_clean])

    prediction = model.predict(text_tfidf)[0]

    if prediction == 1:
        return "spam"
    else:
        return "ham"

# 실행
if __name__ == "__main__":
    while True:
        msg = input("\n텍스트 입력 (종료: exit) : ")

        if msg.lower() == "exit":
            break

        result = predict_spam(msg)
        print("결과 :", result)