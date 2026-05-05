# main.py
# Single entry point — all routes here

import os
import pandas as pd
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_agent
from tools import check_alerts
from gmail import setup_gmail, get_gmail_creds
from memory import save_invoice, get_invoices, get_spend_summary
from config import EXCEL_FILE, UPLOADS_DIR

load_dotenv()

app = FastAPI(title="OpsPilot", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request models ────────────────────────────

class ChatRequest(BaseModel):
    user_name: str
    message: str

class GmailSetupRequest(BaseModel):
    gmail: str
    app_password: str


# ── Pages ─────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ── Health ────────────────────────────────────

@app.get("/health")
async def health():
    gmail, pw = get_gmail_creds()
    return {
        "status": "running",
        "gmail_connected": pw is not None,
        "gmail": gmail if pw else None,
    }


# ── Chat ──────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        response = await run_agent(req.user_name, req.message)
        return {"response": response, "user": req.user_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Suppliers ─────────────────────────────────

@app.get("/suppliers")
def get_suppliers():
    try:
        df = pd.read_excel(EXCEL_FILE, sheet_name="Suppliers")
        return df.fillna("").to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Invoices ──────────────────────────────────

@app.get("/invoices")
def list_invoices():
    try:
        return get_invoices()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-invoice")
async def upload_invoice(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        path = os.path.join(UPLOADS_DIR, file.filename)
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)

        # Try PDF text extraction for vendor/amount
        vendor_name = file.filename.rsplit(".", 1)[0]
        amount      = 0.0

        try:
            import pdfplumber, re
            with pdfplumber.open(path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            lines = text.splitlines()
            if lines:
                vendor_name = lines[0][:80]
            money = re.findall(r"\d+(?:,\d{3})*(?:\.\d{1,2})?", text)
            if money:
                amount = float(money[-1].replace(",", ""))
        except Exception:
            pass

        save_invoice(file.filename, vendor_name, amount)

        return {"message": "Uploaded", "parsed": {"vendor_name": vendor_name, "amount": amount}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Reports ───────────────────────────────────

@app.get("/reports")
def get_reports():
    try:
        data = get_spend_summary()

        # Category breakdown from Excel
        try:
            df = pd.read_excel(EXCEL_FILE, sheet_name="Procurement_History")
            by_cat = (
                df.groupby("Category")["Total_Price_INR"]
                .sum().sort_values(ascending=False).head(5)
            )
            breakdown = "\n".join(f"• {cat}: ₹{int(val):,}" for cat, val in by_cat.items())
        except Exception:
            breakdown = "No procurement history available."

        data["category_breakdown"] = breakdown
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Alerts ────────────────────────────────────

@app.get("/alerts")
def get_alerts():
    try:
        raw = check_alerts()
        # check_alerts returns a string — convert to list of dicts for the frontend
        if isinstance(raw, str):
            lines = [l.strip("• ").strip() for l in raw.strip().splitlines() if l.strip()]
            alerts = [{"level": "High", "message": l} for l in lines]
        else:
            alerts = raw
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Gmail ─────────────────────────────────────

@app.post("/gmail-setup")
def gmail_setup(req: GmailSetupRequest):
    try:
        ok = setup_gmail(req.gmail, req.app_password)
        if ok:
            return {"status": "connected", "gmail": req.gmail}
        raise HTTPException(status_code=400, detail="Gmail connection failed. Check credentials.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Admin ─────────────────────────────────────

@app.get("/rebuild-index")
async def rebuild_index_route():
    try:
        from rag import rebuild_index
        rebuild_index()
        return {"status": "Index rebuilt successfully"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/rag/rebuild")
def rag_rebuild():
    try:
        from rag import rebuild_index
        rebuild_index()
        return {"status": "RAG index rebuilt successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/invalidate")
def cache_invalidate():
    try:
        from retrieval import invalidate_cache
        invalidate_cache()
        return {"status": "Excel cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))