# Nexygen-Net-Zero-Project
This is a time series forecasting project of a power company named Nexygen that is forecasting with the aim of achieving a net zero carbon emission by the year 2040

The Nexygen Net Zero Project is a technical repository centered on time series forecasting for an energy utility company. This initiative aims to assist the organization in reaching a net zero carbon footprint by the target year of 2040. The provided files include Environmental, Social, and Governance (ESG) data, as well as pre-trained machine learning models specifically designed for different emission scopes. The software environment relies on a Python-based stack, incorporating essential libraries such as Pandas for data manipulation and Scikit-learn for predictive modeling. Furthermore, FastAPI, Streamlit, and Docker configurations are included so that the project is production ready for deployment as a web service. Overall, this project demonstrated a data-driven approach to tracking and reducing atmospheric pollutants in the power sector.

# 🌍 NEXYGEN Net-Zero Emissions Forecasting Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

NEXYGEN is a **full-stack, containerized data science application** that forecasts **Scope 1 and Scope 2 carbon emissions** using pre-trained SARIMA models.

It demonstrates:
- End-to-end ML deployment
- API design with FastAPI
- Interactive data apps with Streamlit
- Production-ready containerization with Docker

---

## 🎯 Key Highlights

✅ ML model deployment (Statsmodels - SARIMA)  
✅ RESTful API with auto-generated docs  
✅ Interactive frontend for forecasting  
✅ Fully reproducible with Docker  
✅ Clean modular architecture  

---

## 🧱 Architecture
    ┌────────────────────┐
    │   Streamlit UI     │
    │  (Frontend:8501)   │
    └─────────┬──────────┘
              │ HTTP Requests
              ▼
    ┌────────────────────┐
    │   FastAPI Backend  │
    │   (API:8000)       │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │  SARIMA Models     │
    │ (.pkl files)       │
    └────────────────────┘

    
---

## 📁 Project Structure
docker_practice/
│
├── backend/
│ ├── app.py
│ ├── requirements.txt
│ ├── dockerfile
│ ├── scope1_model.pkl
│ └── scope2_model.pkl
│
├── frontend/
│ ├── streamlit_app.py
│ ├── requirements.txt
│ └── dockerfile.streamlit
│
└── docker-compose.yml


## ⚙️ Tech Stack

| Layer       | Technology |
|------------|-----------|
| Backend     | FastAPI |
| Frontend    | Streamlit |
| ML Models   | Statsmodels (SARIMA) |
| Container   | Docker |
| Orchestration | Docker Compose |
| Data        | Pandas |


## 🚀 Getting Started (Reproducible Setup)

### 🔹 1. Clone the Repository

```bash
git clone <your-repo-url>
cd docker_practice
🔹 2. Build & Run
docker compose up --build

Run in background:

docker compose up --build -d
🔹 3. Access the App
Service	URL
🔧 API Docs (Swagger)	http://127.0.0.1:8000/docs

📊 Frontend App	http://127.0.0.1:8501
🔹 4. Stop Services
docker compose down
🧪 API Usage
POST /forecast
Request
{
  "emission_type": "scope1",
  "steps": 6
}
Response
{
  "emission_type": "scope1",
  "forecast": [123.4, 125.6],
  "dates": ["2024-11-01", "2024-12-01"],
  "last_training_date": "2024-10-01"
}
📸 Screenshots (Add Yours)

💡 Add screenshots here to boost your portfolio impact

🔹 API Docs

🔹 Streamlit Dashboard

🧠 Key Implementation Details
🔹 Model Loading (Startup Lifecycle)

Models are loaded once using FastAPI lifespan events

Improves performance and avoids repeated loading

🔹 Forecast Logic

Uses SARIMA models from Statsmodels

Dynamically generates future monthly timestamps

🔹 Containerization

Separate containers for frontend & backend

Ports exposed:

8000 → API

8501 → UI

⚠️ Known Issues & Fixes
❌ Model Not Found

Ensure:

scope1_model.pkl
scope2_model.pkl

exist inside /backend

❌ Missing Dependency (e.g. pyarrow)

Add to:

backend/requirements.txt

Then rebuild:

docker compose up --build
❌ Frontend Error: requests not defined

Fix:

pip install requests

Add to:

frontend/requirements.txt
❌ Streamlit File Not Found

Ensure Dockerfile uses correct filename:

CMD ["streamlit", "run", "streamlit_app.py"]


🔮 Future Enhancements

🌐 Cloud deployment (AWS / Azure / GCP)

🗄️ Database integration

🔄 Automated model retraining pipeline

🔐 Authentication & user roles

📈 Advanced visual analytics

👨‍💻 Author

Stephen Ogodo
Data Scientist | ML Engineer | FinTech Enthusiast