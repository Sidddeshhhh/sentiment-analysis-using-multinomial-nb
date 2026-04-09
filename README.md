# 🎬 IMDB Sentiment Analysis using Multinomial Naïve Bayes

## 📌 Project Overview
This project performs sentiment analysis on IMDB movie reviews using Natural Language Processing (NLP) techniques. The model classifies reviews as positive or negative using the Multinomial Naïve Bayes algorithm.

---

## 📊 Dataset
- Dataset: IMDB Dataset (50,000 movie reviews)
- Each review is labeled as positive or negative  

🔗 Download dataset from:  
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews  

📌 Note: Place the dataset file (`IMDB Dataset.csv`) in the project folder before running the code.

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NLTK  
- Scikit-learn  
- Seaborn  
- Matplotlib  

---

## 🔄 Workflow
1. Data Loading  
2. Text Cleaning (remove URLs, punctuation, etc.)  
3. Tokenization  
4. Stopword Removal  
5. Stemming using PorterStemmer  
6. Feature Extraction using CountVectorizer  
7. Train-Test Split (80-20)  
8. Model Training (Multinomial Naïve Bayes)  
9. Model Evaluation  

---

## 🤖 Model Used
- Multinomial Naïve Bayes  

---

## 📈 Evaluation Metrics
- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  

---

## 📌 Results
- Model Accuracy: **85%**
- Achieved strong performance in classifying positive and negative reviews  
- Model performs well on unseen text inputs  

---

## 🚀 Key Features
- Implemented complete NLP preprocessing pipeline  
- Used CountVectorizer for text feature extraction  
- Built a custom prediction function for new reviews  
- Visualized confusion matrix using heatmap  

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/Siddeshhhh/sentiment-analysis-using-multinomial-nb.git
cd sentiment-analysis-using-multinomial-nb
