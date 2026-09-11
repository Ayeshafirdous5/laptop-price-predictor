# 💻 Laptop Price Predictor

A Machine Learning web application that predicts laptop prices based on various specifications such as **company, laptop type, RAM, CPU, GPU, screen resolution, and storage**.

The application uses a trained **Random Forest Machine Learning model** to generate price predictions. The web interface is built using **Flask and HTML**.

---

## 📌 Project Overview

Laptop prices depend on multiple hardware specifications, making it difficult to estimate the price manually.

This project uses Machine Learning to analyze laptop specifications and predict an estimated laptop price based on the given input.

The application provides a simple web interface where users can enter laptop specifications and receive a predicted price.

---

## ✨ Features

- 💻 Predict laptop prices based on specifications
- 🤖 Machine Learning model using Random Forest
- 🌐 Simple web interface using Flask
- 📝 User-friendly input fields and dropdown menus
- 📊 Handles multiple laptop specifications
- 🖥️ Calculates PPI based on screen resolution and screen size
- 🔄 Includes a Clear button to reset the form
- ⚠️ Handles invalid or missing input gracefully

---

## 🧠 Machine Learning Model

The application uses a **Random Forest Regression model** trained on laptop data.

The model analyzes different specifications and predicts the approximate laptop price.

### Model Used

- Random Forest Regressor

---

## 📊 Input Features

The model uses the following laptop specifications:

- Company
- TypeName
- Inches
- ScreenResolution
- Touchscreen
- IPS
- PPI
- CPU
- RAM
- GPU
- Operating System
- Weight
- HDD
- SSD
- Hybrid Storage
- Flash Storage

---

## 📂 Project Structure

```text
laptop-price-predictor/
│
├── main.ipynb
├── templates/
│   └── index.html
├── app.py
├── laptop_data.csv
├── laptop_price_model.pkl
├── requirements.txt
└── README.md
```
## 📄 File Description

### 📓 main.ipynb

Contains the Machine Learning workflow, including:

- Data loading
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Saving the trained model

### 🐍 app.py

The main Flask application file.

It handles:

- User input
- Data preprocessing
- Feature preparation
- Loading the trained model
- Price prediction
- Displaying results

### 📊 laptop_data.csv

Contains the dataset used for training the Machine Learning model.

### 🤖 laptop_price_model.pkl

Contains the trained Machine Learning model used for predicting laptop prices.

### 🌐 templates/index.html

Contains the frontend interface of the application.

Users can enter laptop specifications and get a predicted price.

### 📦 requirements.txt

Contains the Python libraries required to run the project.

---

## ⚙️ How It Works

```text
Laptop Specifications
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Trained Random Forest Model
        ↓
Laptop Price Prediction
```

---

## 🖥️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Random Forest

### Data Analysis

- Pandas
- NumPy

### Web Framework

- Flask

### Frontend

- HTML
- CSS

### Development Environment

- Jupyter Notebook
- Visual Studio Code

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Ayeshafirdous5/laptop-price-predictor.git
```

### 2. Navigate to the Project Folder

```bash
cd laptop-price-predictor
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application

```bash
python app.py
```

### 5. Open in Browser

After running the application, open:

```text
http://127.0.0.1:5000/
```

---

## 📦 Requirements

Make sure Python is installed on your system.

Recommended version:

- Python 3.9+

Required libraries include:

- Flask
- Pandas
- NumPy
- Scikit-learn
- Pickle

---

## 📈 Future Improvements

The project can be further improved by adding:

- 📊 Model performance comparison
- 🤖 Multiple Machine Learning models
- 🌐 Improved user interface
- 📱 Responsive design
- ☁️ Cloud deployment
- 📉 Data visualization dashboard
- 🔍 More detailed laptop specifications
- ⭐ Model accuracy improvements

---

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Python Programming
- Data Preprocessing
- Feature Engineering
- Machine Learning
- Random Forest Model
- Model Serialization
- Flask
- Web Development
- HTML
- Building an end-to-end Machine Learning application

---

## 👩‍💻 Author

**Ayesha Firdous**

Computer Science Engineering Student

---

## ⭐ Conclusion

This project demonstrates how Machine Learning can be integrated with a web application to solve a real-world problem.

The trained model analyzes laptop specifications and provides an estimated price prediction through an easy-to-use Flask web interface.
