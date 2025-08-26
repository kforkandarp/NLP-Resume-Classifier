# 📄 NLP Resume Screening Project

An **NLP-powered Resume Classifier** that helps recruiters automatically screen resumes and categorize them into different job roles using machine learning models and a simple Flask web app.

---

## 🔥 Features
- 📂 **Resume Upload** – Upload PDF resumes for classification.  
- 🧠 **NLP Model** – Pre-trained ML models classify resumes into roles (e.g., Data Science, Web Development, HR, etc.).  
- 📊 **Dataset** – Uses the [Updated Resume Dataset](UpdatedResumeDataSet.csv).  
- 🌐 **Flask Web App** – User-friendly interface to upload resumes and view predictions.  
- 📓 **Jupyter Notebook** – Training and analysis provided in `Resume Screening with Python.ipynb`.  

---

## 📊 Dataset
- **Source**: Updated Resume Dataset (`UpdatedResumeDataSet.csv`).  
- Contains resumes categorized into different job roles.  
- Used for training ML models like Logistic Regression, SVM, or Random Forest with TF-IDF Vectorization.  

---

## 🧠 NLP & ML Pipeline
1. **Data Cleaning** – Removing stopwords, punctuation, and special characters.  
2. **Feature Extraction** – TF-IDF vectorization of resume text.  
3. **Model Training** – Logistic Regression, SVM, or Random Forest classifiers.  
4. **Model Saving** – Models saved as `.pkl` files for reuse.  
5. **Prediction** – Flask app loads trained model and classifies uploaded resumes.  

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/NLP-Resume-Classifier.git
cd NLP-Resume-Classifier
````

### 2️⃣ Create & activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask app

```bash
python app.py
```

Visit **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your browser 🎉

---

## 📂 Project Structure

```
NLP-Resume-Classifier/
│── app.py                        # Flask web app
│── requirements.txt              # Dependencies
│── UpdatedResumeDataSet.csv      # Dataset
│── Resume Screening with Python.ipynb  # Training notebook
│── static/                       # CSS/JS files
│── templates/                    # HTML templates
│── models/                       # Saved ML models (.pkl)
```

---


