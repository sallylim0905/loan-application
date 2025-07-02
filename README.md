# 🏦 Loan Application Prediction System

A simple web-based machine learning application to predict loan approval using a logistic regression model. Built with Flask and Scikit-learn.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 📝 User-friendly loan application form  
- 🤖 Real-time loan approval prediction using machine learning  
- 🖥️ Simple and intuitive web interface  
- 🔁 End-to-end ML pipeline: training → deployment → inference  

---

## 🧰 Tech Stack

- **Backend**: Python, Flask  
- **Machine Learning**: scikit-learn, NumPy, Pandas  
- **Frontend**: HTML, CSS (Bootstrap)  
- **Model Saving**: joblib / pickle

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.x  
- Git  

---

### 📥 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sallylim0905/loan-application.git
   cd loan-application
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage

### Step 1: Train the ML Model

```bash
python train_model.py
```

> Make sure the dataset file `DS_Capstone_DataSet -REV.csv` is in the project folder.

### Step 2: Launch the Web App

```bash
python app.py
```

Then open your browser and go to:  
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📁 Project Structure

```
loan-application/
├── app.py                      # Flask web app
├── train_model.py              # ML training script
├── logistic_regression_model.pkl  # Saved ML model
├── requirements.txt
├── DS_Capstone_DataSet -REV.csv   # Dataset file
└── templates/
    └── index.html              # Frontend HTML form
```

---

## 🤝 Contributing

1. Fork the repository  
2. Create a new branch:  
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit:  
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:  
   ```bash
   git push origin feature-branch
   ```
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

Developed by [Sally Lim](https://github.com/sallylim0905)


