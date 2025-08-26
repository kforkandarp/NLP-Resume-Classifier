# 📄 NLP Resume Screening Project

This project is an **NLP-powered Resume Classifier** that automatically screens resumes and classifies them into different job categories.  
It helps recruiters quickly identify suitable candidates by analyzing resumes and matching them with predefined roles.

---

## 🔥 Features
- 📂 **Resume Upload**: Upload PDF resumes for classification.
- 🧠 **NLP Model**: Pre-trained machine learning models classify resumes into categories (e.g., Data Science, Web Development, HR, etc.).
- 📊 **Dataset**: Uses the [Updated Resume Dataset](UpdatedResumeDataSet.csv).
- 🌐 **Flask Web App**: Simple web interface for uploading resumes and viewing predictions.
- 📓 **Jupyter Notebook**: Includes the notebook (`Resume Screening with Python.ipynb`) for training and analysis.

---
📊 Dataset

The project uses the Updated Resume Dataset (UpdatedResumeDataSet.csv).

Contains resumes categorized into different job roles.

Used for training ML models like Logistic Regression, SVM, or Random Forest with TF-IDF Vectorization.

🧠 NLP & ML Pipeline

Data Cleaning → Removing stopwords, punctuation, and special characters.

Feature Extraction → TF-IDF vectorization of text.

Model Training → ML classifier (e.g., Logistic Regression, SVM).

Model Saving → Models stored as .pkl files for reuse in the web app.

Prediction → Flask app loads trained model and classifies uploaded resumes.