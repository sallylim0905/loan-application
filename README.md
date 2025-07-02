
---

````markdown
# 🏦 Loan Application Prediction System

A simple web-based machine learning application to predict loan approval using a logistic regression model. Built with Flask and Scikit-learn.

## ✨ Features
- User-friendly loan application form with input validation  
- Real-time loan approval prediction  
- Simple and intuitive UI  
- End-to-end ML workflow (train, deploy, predict)

## 🧰 Tech Stack
- **Backend**: Python, Flask  
- **ML Framework**: scikit-learn, NumPy, Pandas  
- **Frontend**: HTML, CSS, Bootstrap (via `index.html`)  
- **Model Serialization**: joblib / pickle

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python 3.x  
- Git  

### 📥 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sallylim0905/loan-application.git
   cd loan-application
````

2. **Set up a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage

### Step 1: Train the Model

Run the training script to generate `logistic_regression_model.pkl`:

```bash
python train_model.py
```

> Make sure the dataset (`DS_Capstone_DataSet -REV.csv`) is in the project directory.

### Step 2: Launch the Web App

```bash
python app.py
```

Navigate to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## 📁 File Structure

```
loan-application/
├── app.py                      # Flask web application
├── train_model.py              # Model training script
├── logistic_regression_model.pkl  # Saved trained model
├── requirements.txt
├── DS_Capstone_DataSet -REV.csv   # Dataset file
└── templates/
    └── index.html              # Frontend form
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch:

   ```bash
   git checkout -b feature-branch
   ```
3. Make changes and commit:

   ```bash
   git commit -m "Add feature"
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

## 📬 Contact

Developed by [Sally Lim](https://github.com/sallylim0905)

---

