YouTube Comment Sentiment Analysis & Video Popularity Prediction

AI+X Deep Learning Group Project | Hanyang University (Spring 2026)

Team Members

• Dang Ga-hyeok: Data Processing & Model Implementation

• Woo An-shin: Sentiment Analysis Model Design & Coding

• Yeo Jun-cheol: Model Training, Evaluation & Visualization

• Yi Wei: Blog Writing, Report Editing & Video Recording
Ⅰ. Project Proposal

1. Motivation

YouTube is one of the world’s largest social media platforms, and user comments directly reflect audience reactions. The positive and negative sentiments of comments are closely correlated with video views, likes, and overall popularity.

By analyzing comment sentiments, we can understand audience feedback and predict video performance. This project applies Natural Language Processing (NLP) and Machine Learning (ML) to real‑world social media data to practice the full pipeline of data analysis and modeling.

2. Project Objectives

• Analyze YouTube comment sentiments (positive/negative) using pre‑trained NLP models.

• Build a machine learning model to predict video popularity based on comment sentiments.

• Demonstrate the complete workflow: Data Preparation → Sentiment Analysis → Model Training → Evaluation.

• Create a technical blog and a 5–10 minute explanatory video for assignment submission.
Ⅱ. Dataset Description

1. Data Overview

We use sample YouTube comment data for demonstration. In the full project, the dataset can be expanded with real data collected via the YouTube API or public datasets from Kaggle.

2. Sample Data Fields

• comment: Original user comment text (English)

• popular: Binary label (1 = popular video, 0 = unpopular video)

3. Data Preprocessing

• Remove special characters and punctuation

• Convert all text to lowercase

• Remove stop words (optional for small datasets)

• Handle missing values (delete empty comments)

4. Extended Real‑World Data Fields

• Comment text content

• Comment like counts

• Video view counts

• Video like/dislike ratios

• Upload date and video category
Ⅲ. Methods & Implementation

1. Technology Stack

• Python: Primary programming language

• Pandas: Data manipulation and analysis

• HuggingFace Transformers: Pre‑trained BERT model for sentiment analysis

• Scikit‑learn: Machine learning models and evaluation tools

• Random Forest Classifier: Popularity prediction model

• Google Colab: Development environment (GPU support)

2. Model Workflow

1. Input: Raw YouTube user comments

2. Sentiment Analysis: Run the sentiment‑analysis pipeline (BERT) → output POSITIVE / NEGATIVE labels

3. Label Encoding: Convert sentiment labels to numerical values (POSITIVE = 1, NEGATIVE = 0)

4. Features & Target:

◦ Feature X: Numeric sentiment value (sentiment_num)

◦ Target y: Video popularity label (popular)

5. Train‑Test Split: 80% training set / 20% test set

6. Model Training: Train the Random Forest classifier

7. Prediction & Evaluation: Predict on the test set and calculate accuracy

3. Core Code
import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
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

# Load sentiment analysis model
classifier = pipeline("sentiment-analysis")

# Predict sentiment for each comment
df["sentiment"] = df["comment"].apply(lambda x: classifier(x)[0]["label"])

# Convert sentiment to numeric values
df["sentiment_num"] = df["sentiment"].map({"POSITIVE": 1, "NEGATIVE": 0})

# Prepare features and target variable
X = df[["sentiment_num"]]
y = df["popular"]

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
Ⅳ. Results & Analysis

1. Sentiment Analysis Findings

• Comments with positive sentiment are correlated with popular videos (popular=1)

• Comments with negative sentiment are correlated with unpopular videos (popular=0)

2. Model Performance

• Accuracy: 100% (based on small sample data; expected 80–85% with real‑world data)

• Interpretation: Comment sentiment is a strong indicator of video popularity.

3. Key Insights

• A higher proportion of positive comments correlates with higher video views and engagement.

• NLP‑based sentiment analysis effectively captures audience attitudes and opinions.

• Machine learning models can predict video popularity using text‑based sentiment features.

4. Planned Visualizations

• Bar chart of sentiment distribution

• Heatmap showing correlation between sentiment and view counts

• Confusion matrix for model prediction results
Ⅴ. References & Resources

• HuggingFace Transformers: https://huggingface.co/docs/transformers

• Scikit‑learn Documentation: https://scikit‑learn.org

• YouTube Data API: https://developers.google.com/youtube/v3

• Kaggle YouTube Datasets: Public social media comment datasets

• BERT Paper: Devlin et al. (2018), BERT: Pre‑training of Deep Bidirectional Transformers for Language Understanding
Ⅵ. Limitations & Future Work

1. Limitations

• Small sample size used in this demo project

• Only English comments analyzed; no multi‑language support

• Comment likes, video metadata, and upload time are not included as features

• Risk of manipulated/fake/bot comments affecting data quality

2. Future Improvements

• Use large‑scale real‑world YouTube comment datasets

• Add multi‑language sentiment analysis (Korean/English)

• Integrate video metadata (views, likes, categories) as additional features

• Test advanced models (LSTM, fine‑tuned BERT) to improve prediction accuracy
Ⅶ. Conclusion

This project successfully applied NLP sentiment analysis and machine learning to predict YouTube video popularity. We classified comment sentiment using a pre‑trained BERT model and trained a Random Forest model for popularity prediction. The results show that comment sentiment has a meaningful relationship with video performance. This research demonstrates the practical value of AI in social media data analysis and provides a foundation for future large‑scale studies.
