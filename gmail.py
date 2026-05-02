# gmail.py
# Gmail send + inbox scan

import os
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
    gmail       = get_credential("gmail") or SENDER_EMAIL
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

        conn = get_db()
        conn.execute(
            "INSERT INTO email_log (supplier_name, supplier_email, item_name, status) VALUES (?, ?, ?, ?)",
            (to_name, to_email, subject, "sent"),
        )
        conn.commit()
        conn.close()

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
            msg  = email.message_from_bytes(msg_data[0][1])
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()[:500]
                        break
            else:
                body = msg.get_payload(decode=True).decode()[:500]
            results.append({"from": msg["from"], "subject": msg["subject"], "preview": body})
        mail.logout()
        save_credential("last_scanned", datetime.now().isoformat())
        if not results:
            return "No new emails found."
        return "\n\n".join(f"{i+1}. From: {r['from']}\nSubject: {r['subject']}\n{r['preview'][:200]}"
                           for i, r in enumerate(results))
    except Exception as e:
        return f"❌ Scan failed: {e}"