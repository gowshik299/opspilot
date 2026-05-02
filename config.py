# config.py
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
STATIC_DIR  = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

EXCEL_FILE  = os.path.join(DATA_DIR, "procurement_data.xlsx")
DB_PATH     = os.path.join(BASE_DIR, "opspilot.db")
RAG_STORE   = os.path.join(DATA_DIR, "rag_store.pkl")

PDF_FILES = [
    "safety_manual.pdf",
    "outage_procedures.pdf",
    "equipment_maintenance.pdf",
]

for d in [DATA_DIR, STATIC_DIR, UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)