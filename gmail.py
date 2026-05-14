# gmail.py
# Gmail send + inbox scan

import os
import re
import json
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

from memory import save_credential, get_credential, get_db

load_dotenv()

SENDER_EMAIL = os.getenv("GMAIL", "")


def clean_email_body(body: str) -> str:
    """Strip HTML tags and clean email body thoroughly"""
    # Remove style and script blocks completely
    body = re.sub(r'<style[^>]*>.*?</style>', '', body, flags=re.DOTALL)
    body = re.sub(r'<script[^>]*>.*?</script>', '', body, flags=re.DOTALL)
    # Remove HTML tags
    body = re.sub(r'<[^>]+>', ' ', body)
    # Decode HTML entities
    body = body.replace('&nbsp;', ' ')
    body = body.replace('&amp;', '&')
    body = body.replace('&lt;', '<')
    body = body.replace('&gt;', '>')
    body = body.replace('&#39;', "'")
    body = body.replace('&quot;', '"')
    # Remove base64 chunks
    body = re.sub(r'[A-Za-z0-9+/]{50,}={0,2}', '', body)
    # Remove URLs
    body = re.sub(r'https?://\S+', '', body)
    # Remove email headers leaked into body
    body = re.sub(r'(From|To|Subject|Date|MIME|Content):[^\n]*\n', '', body)
    # Remove extra whitespace
    body = re.sub(r'\s+', ' ', body).strip()
    return body


def setup_gmail(gmail: str, app_password: str) -> bool:
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(gmail, app_password)
        server.quit()
        save_credential("gmail", gmail)
        save_credential("app_password", app_password)
        return True
    except Exception as e:
        print(f"Gmail setup error: {e}")
        return False


def get_gmail_creds() -> tuple:
    gmail        = get_credential("gmail") or SENDER_EMAIL
    app_password = get_credential("app_password")
    return gmail, app_password


def send_email(to_email: str, to_name: str, subject: str, body: str) -> str:
    gmail, app_password = get_gmail_creds()
    if not app_password:
        return "❌ Gmail not connected. Use Gmail Setup first."
    try:
        msg = MIMEMultipart()
        msg["From"]    = gmail
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(gmail, app_password)
        server.sendmail(gmail, to_email, msg.as_string())
        server.quit()

        from sqlalchemy import text
        with get_db() as conn:
            conn.execute(
                text("INSERT INTO email_log (supplier_name, supplier_email, item_name, status) VALUES (:a, :b, :c, :d)"),
                {"a": to_name, "b": to_email, "c": subject, "d": "sent"},
            )
            conn.commit()

        return f"✅ Email sent to {to_name}"
    except Exception as e:
        return f"❌ Failed: {e}"


def scan_inbox(last_n: int = 10) -> str:
    gmail, app_password = get_gmail_creds()
    if not app_password:
        return "❌ Gmail not connected."
    try:
        last_scanned = get_credential("last_scanned")
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(gmail, app_password)
        mail.select("inbox")
        since = (datetime.fromisoformat(last_scanned).strftime("%d-%b-%Y")
                 if last_scanned else datetime.now().strftime("%d-%b-%Y"))
        _, messages = mail.search(None, f'(SINCE "{since}")')
        ids = messages[0].split()[-last_n:]
        results = []

        for eid in ids:
            _, msg_data = mail.fetch(eid, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])

            plain_body = ""

            if msg.is_multipart():
                # First try plain text
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        raw = part.get_payload(decode=True).decode(errors='ignore')
                        plain_body = clean_email_body(raw)
                        break
                # Fallback to HTML
                if not plain_body:
                    for part in msg.walk():
                        if part.get_content_type() == "text/html":
                            raw = part.get_payload(decode=True).decode(errors='ignore')
                            plain_body = clean_email_body(raw)
                            break
            else:
                raw = msg.get_payload(decode=True).decode(errors='ignore')
                plain_body = clean_email_body(raw)

            # Clean subject (sometimes encoded)
            subject = msg["subject"] or "No subject"
            sender  = msg["from"] or "Unknown"

            results.append({
                "from":    sender,
                "subject": subject,
                "preview": plain_body[:120],
                "full":    plain_body
            })

        mail.logout()
        save_credential("last_scanned", datetime.now().isoformat())

        if not results:
            return "No new emails found."

        return json.dumps(results)

    except Exception as e:
        return f"❌ Scan failed: {e}"