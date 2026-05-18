# scheduler.py
import logging
import json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()
latest_notifications = []


async def check_supplier_emails():
    """Check Gmail for new supplier emails every 2 minutes"""
    try:
        from gmail import scan_inbox, get_gmail_creds
        gmail, password = get_gmail_creds()
        if not password:
            return

        result = scan_inbox(last_n=5)
        if result.startswith("❌") or result == "No new emails found.":
            return

        emails = json.loads(result)
        supplier_domains = ["bel.in", "pgc.in", "safeguard.in", "vcpl.in",
                           "nmw.in", "sparkrelay.in", "atlas.in", "insupower.in"]

        for email in emails:
            sender = email.get("from", "").lower()
            if any(domain in sender for domain in supplier_domains):
                latest_notifications.append({
                    "type": "email",
                    "title": "Supplier Email Received!",
                    "message": f"From: {email['from']}\nSubject: {email.get('subject','')}",
                    "preview": email.get("preview", "")
                })
                logger.info(f"Supplier email detected: {email.get('subject')}")

    except Exception as e:
        logger.error(f"Email check error: {e}")


async def check_new_alerts():
    """Check for new high priority alerts every 5 minutes"""
    try:
        from tools import query_db
        rows = query_db("""
            SELECT item_name, priority, status
            FROM pending_requirements
            WHERE LOWER(priority) = 'high'
            AND LOWER(status) = 'open'
            ORDER BY id DESC
            LIMIT 3
        """)
        if rows:
            if not any(n.get("type") == "alert" for n in latest_notifications):
                latest_notifications.append({
                    "type": "alert",
                    "title": f"{len(rows)} High Priority Alerts!",
                    "message": "\n".join(f"• {r['item_name']}" for r in rows)
                })
    except Exception as e:
        logger.error(f"Alert check error: {e}")


def start_scheduler():
    scheduler.add_job(check_supplier_emails, IntervalTrigger(minutes=2), id="email_checker", replace_existing=True)
    scheduler.add_job(check_new_alerts, IntervalTrigger(minutes=5), id="alert_checker", replace_existing=True)
    scheduler.start()
    logger.info("✅ Scheduler started!")


def stop_scheduler():
    scheduler.shutdown()


def get_notifications():
    global latest_notifications
    notifs = latest_notifications.copy()
    latest_notifications = []
    return notifs