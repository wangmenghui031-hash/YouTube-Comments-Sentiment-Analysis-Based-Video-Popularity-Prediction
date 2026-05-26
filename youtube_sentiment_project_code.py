import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 샘플 데이터
data = {
    "comment": [
        "This video is amazing!",
        "I hate this content",
        "Very helpful and interesting",
        "Terrible video"
    ],
    "popular": [1, 0, 1, 0]
}

df = pd.DataFrame(data)

# 감성 분석 모델 로드
classifier = pipeline("sentiment-analysis")

# 감성 분석 수행
df["sentiment"] = df["comment"].apply(lambda x: classifier(x)[0]["label"])

# 숫자 변환
df["sentiment_num"] = df["sentiment"].map({"POSITIVE":1, "NEGATIVE":0})

X = df[["sentiment_num"]]
y = df["popular"]

# 학습 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 예측
pred = model.predict(X_test)

# 정확도 출력
print("Accuracy:", accuracy_score(y_test, pred))
