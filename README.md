# AI-Based Data Quality Monitoring System

This project is an **end-to-end AI-based data quality monitoring application** that combines  
**cloud machine learning predictions** with **statistical data analysis and interactive dashboards** in a single Streamlit web app.

The system helps identify potential data quality issues by using:
- Real-time AI predictions from a cloud-deployed ML model
- Data cleaning and anomaly detection on historical datasets
- Visual dashboards for easy interpretation of data health

<img width="1850" height="762" alt="image" src="https://github.com/user-attachments/assets/b3eb0a13-800c-4dbc-8936-39bc1dd9db4b" />
<img width="1919" height="935" alt="image" src="https://github.com/user-attachments/assets/0b025f12-64b9-4dbd-b1cd-e03ff8de360f" />

---

## 🚀 Key Features

- Real-time **data quality prediction** using a cloud ML model
- Secure **REST API integration** with token-based authentication
- Data cleaning (missing values, duplicates, formatting)
- **Anomaly detection** using statistical methods (Z-score)
- Interactive **dashboard with metrics, plots, and tables**
- Single unified Streamlit application
- Cloud-ready deployment using Docker

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Frontend & UI:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Plotly  
- **Machine Learning:** Cloud-deployed ML model (via REST APIs)  
- **Cloud & Deployment:** Docker, Hugging Face Spaces  

---

## 📂 Project Structure  
aipredict-space  
├── aipredict.py  
├── requirements.txt  
└── README.md  
└── orders_data_set_final.csv

## ⚙️ How the Project Works (High-Level)

1. User enters raw order data in the web UI
2. The app authenticates with the cloud ML service
3. Input data is sent to a deployed ML model via REST APIs
4. The model returns a prediction about data quality issues
5. A CSV dataset is loaded for historical analysis
6. Data is cleaned and validated
7. Anomalies are detected using statistical methods
8. Results are visualized using interactive dashboards

---

## 🧩 Application Flow (Detailed)

### 1️⃣ Prediction Module
- Accepts user inputs such as order details and customer information
- Formats data into a structured JSON payload
- Sends the payload to a cloud ML deployment using REST APIs
- Displays the predicted data quality issue in real time

### 2️⃣ Data Cleaning Module
- Converts date columns into proper formats
- Handles missing values and removes duplicates
- Ensures consistent and reliable data for analysis

### 3️⃣ Anomaly Detection
- Uses Z-score based statistical analysis
- Flags unusual or extreme values as anomalies
- Adds an `anomaly` indicator column to the dataset

### 4️⃣ Dashboard & Visualization
- Displays key metrics such as total records and anomaly count
- Uses box plots to visualize data distribution and outliers
- Shows detected anomalies in a tabular format for inspection

---  

## 🎯 Use Cases  
-Monitoring data quality in data pipelines  
- Detecting anomalies in transactional datasets
- Supporting data engineering and analytics workflows
- Demonstrating real-world ML model integration

---  

## 🧠 Learning Outcomes  
This project demonstrates:  
- End-to-end ML system integration
- REST API based model consumption
- Secure credential management
- Data preprocessing and anomaly detection
- Interactive analytics dashboards
- Cloud deployment and debugging
