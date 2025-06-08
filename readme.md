ni225-streamlit-app/
├── app.py                    # Main Streamlit application
├── ni225_model_1.pkl         # First trained model (binary or multi-class)
├── ni225_model_2.pkl         # Second trained model
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image configuration
└── README.md                 # Optional: explanation and setup guide


# In the root directory (ni225-streamlit-app/)
docker build -t ni225-streamlit-app .
docker run -p 8501:8501 ni225-streamlit-app

GCP_PROJECT	sample-335613	
TABLE_ID	database_stock	
SHEET_ID	ManualOperation_NI225_DiffRFModel