# 🧠 Mental Health Prediction Using Social Media Text

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify whether a user's social media post indicates signs of depression. It combines sentiment analysis and model training to provide real-time predictions via a web interface.

---

## 📌 Problem Statement

Social media platforms contain vast amounts of text that can reflect users’ mental health conditions. This project aims to predict whether a post indicates depressive symptoms, helping support early mental health detection using machine learning.

---

## 💡 Features

- Sentiment analysis using `TextBlob`
- TF-IDF vectorization for feature extraction
- Model training with **KNN**, **Logistic Regression**, and **Random Forest**
- Evaluation using **AUC**, **F1-score**, **ROC Curve**, and **Cross-Validation**
- Real-time prediction via **Streamlit** web app

---

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, TextBlob, Matplotlib, Seaborn
- **NLP**: TF-IDF Vectorizer, Sentiment Analysis
- **Web App**: Streamlit
- **Deployment**: Localhost (Streamlit UI)

---

## 🚀 Project Structure

```
📦 Mental-Health-Prediction/
├── 📁 venv/                # Virtual environment (excluded via .gitignore)
├── 📄 pipeline.py          # Model training + saving
├── 📄 app.py               # Streamlit frontend app
├── 📄 depressionc.json     # Twitter data (pre-collected)
├── 📄 model.pkl            # Trained RandomForest model
├── 📄 vectorizer.pkl       # Saved TF-IDF vectorizer
├── 📄 requirements.txt     # Python dependencies
├── 📄 README.md            # Project description
└── 📄 .gitignore           # Ignored files/folders
```

---

## 📊 Model Performance

| Model               | Accuracy | AUC Score |
|--------------------|----------|-----------|
| K-Nearest Neighbors| ~85%     | 0.85      |
| Logistic Regression| ~89%     | 0.95      |
| 🌟 Random Forest    | **~89%** | **0.946** |

> 🔍 Random Forest performed best and was deployed in production.

---

## 🎯 How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/abubakar07g/Mental-Health-Prediction.git
cd Mental-Health-Prediction
```

2. **Set up virtual environment**

```bash
python -m venv venv
venv\Scripts\activate       # For Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run backend pipeline**

```bash
python pipeline.py
```

5. **Run frontend Streamlit app**

```bash
streamlit run app.py
```

---

## ✨ Sample Output

> **Input:** `"I feel so low today. Nothing makes sense anymore."`  
> **Output:** `Prediction: Depressed`

> **Input:** `"Had an amazing time with family!"`  
> **Output:** `Prediction: Not Depressed`

---

## 👨‍💻 Author

- **Mohammad Abubakar** — [@abubakar07g](https://github.com/abubakar07g)

---

⭐ *Star this repo if you found it helpful or inspiring!*
