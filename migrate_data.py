import pandas as pd
import os
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL")
EXCEL_FILE = "data/procurement_data.xlsx"

engine = create_engine(DATABASE_URL)

def migrate_suppliers():
    print("Migrating suppliers...")
    df = pd.read_excel(EXCEL_FILE, sheet_name="Suppliers")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    print(f"Columns: {df.columns.tolist()}")
    df.to_sql("suppliers", engine, if_exists="append", index=False)
    print(f"✅ Migrated {len(df)} suppliers")

def migrate_procurement():
    print("Migrating procurement history...")
    df = pd.read_excel(EXCEL_FILE, sheet_name="Procurement_History")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    print(f"Columns: {df.columns.tolist()}")
    df.to_sql("procurement_history", engine, if_exists="append", index=False)
    print(f"✅ Migrated {len(df)} procurement records")

def migrate_pending():
    print("Migrating pending requirements...")
    df = pd.read_excel(EXCEL_FILE, sheet_name="Pending_Requirements")
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    print(f"Columns: {df.columns.tolist()}")
    df.to_sql("pending_requirements", engine, if_exists="append", index=False)
    print(f"✅ Migrated {len(df)} pending requirements")

if __name__ == "__main__":
    migrate_suppliers()
    migrate_procurement()
    migrate_pending()
    print("✅ All data migrated!")
EOF