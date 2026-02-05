# 📧 Email Spam Detector (AI Powered)

A Machine Learning-based web application that predicts whether an incoming email/message is **Spam** or **Ham (Legitimate)**. Built with Python, Flask, and Scikit-learn.



## 🚀 Live Demo
Check out the live app here: [https://email-spam-detector-vr63.onrender.com](https://email-spam-detector-vr63.onrender.com)

## ✨ Features
* **High Accuracy:** 98.7% accuracy using Multinomial Naive Bayes.
* **Real-time Prediction:** Instant results as soon as you hit the "Predict" button.
* **Responsive UI:** Clean and modern interface built with HTML/CSS.
* **Web Hosting:** Deployed on Render for global access.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Framework:** Flask
* **Machine Learning:** Scikit-learn, Pandas, Numpy, NLTK
* **Deployment:** Gunicorn, Render
* **Frontend:** HTML5, CSS3

## 📂 Project Structure
```text
├── models/
│   ├── model.pkl        # Trained Naive Bayes model
│   └── vectorizer.pkl   # TF-IDF Vectorizer
├── templates/
│   └── index.html       # Web Interface
├── app.py               # Flask Application Server
├── requirements.txt     # Python Dependencies
└── README.md            # Project Documentation
