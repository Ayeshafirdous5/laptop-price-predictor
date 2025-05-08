# 💻 Laptop Price Predictor (Web App)

A machine learning web application that predicts laptop prices based on specifications like RAM, CPU, GPU, screen resolution, and storage. Built using **Flask** for the backend and **HTML/CSS (Jinja2 templates)** for the frontend, the app loads a trained Random Forest model to make real-time predictions.

## 📂 Project Structure
LaptopPricePrediction/
├── app.py # Flask app for handling routes and prediction
├── templates/
│ └── index.html # HTML template with form and result rendering
├── laptop_price_model.pkl # Trained RandomForest model
├── laptop_data.csv # Dataset used for model training
├── requirements.txt # Required Python packages
├── README.md # Project overview and instructions

## 📌 Features
- Simple and clean **web interface** using HTML + CSS
- Real-time laptop price prediction using a trained ML model
- Dropdowns and input fields for laptop specs (Company, CPU, RAM, etc.)
- Categorical features encoded with custom mappings
- Automatically calculates **PPI** from resolution and screen size
- Includes a “Clear” button to reset the form
- Handles invalid or missing input gracefully

## 📊 Input Features Used for Prediction

The model was trained using these **15 features**:

- `Company`
- `TypeName`
- `Inches`
- `ScreenResolution`
- `Touchscreen`
- `IPS`
- `PPI` (calculated from resolution and inches)
- `Cpu`
- `Ram`
- `Gpu`
- `OpSys`
- `Weight`
- `HDD`
- `SSD`
- `Hybrid` and `Flash Storage`


## 🛠 How to Run the App

1. **Clone this repository** or download the project ZIP.
2. Make sure you have Python **3.9+** installed.
3. Open a terminal in the project folder and install the dependencies:

```bash
pip install -r requirements.txt

Run the Flask application:
bash: (in the terminal)
python app.py


Open your browser and visit:
like this http://127.0.0.1:5000/ 


✅ Requirements
Python 3.9+
Flask
pandas
numpy
scikit-learn
pickle






📬 Contact
For queries or feedback, contact: ayeshafirdous05@gmail.com

