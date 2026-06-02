# ragas_test_dataset.py
# Generated from actual OpsPilot PDF documents:
# - safety_manual.pdf (Southern Power Distribution Company)
# - equipment_maintenance.pdf
# - outage_procedures.pdf

TEST_DATA = [
    # ── SAFETY MANUAL QUESTIONS ──────────────────────────────
    {
        "question": "What PPE is required for high voltage work?",
        "ground_truth": "All workers must wear approved PPE including safety helmets (Class E), high voltage insulated gloves, safety shoes with steel toe caps, and flame resistant clothing when working near energized equipment.",
        "source": "safety_manual.pdf"
    },
    {
        "question": "What are the general safety rules for working near high voltage equipment?",
        "ground_truth": "All personnel working near high voltage equipment must follow mandatory safety rules at all times. Failure to comply may result in serious injury or death. Workers must wear approved PPE including Class E helmets, insulated gloves, steel toe safety shoes, and flame resistant clothing.",
        "source": "safety_manual.pdf"
    },

    # ── EQUIPMENT MAINTENANCE QUESTIONS ─────────────────────
    {
        "question": "How often should transformer oil level be checked?",
        "ground_truth": "Transformer oil level should be checked weekly by a Junior Engineer.",
        "source": "equipment_maintenance.pdf"
    },
    {
        "question": "How often should transformer oil temperature be checked?",
        "ground_truth": "Transformer oil temperature should be checked daily by an Operator.",
        "source": "equipment_maintenance.pdf"
    },
    {
        "question": "How often should transformer oil dielectric strength be tested?",
        "ground_truth": "Transformer oil dielectric strength test must be conducted every 6 months by a Senior Engineer.",
        "source": "equipment_maintenance.pdf"
    },
    {
        "question": "How often should Buchholz relay be tested?",
        "ground_truth": "Buchholz relay test must be conducted annually by a Certified Technician.",
        "source": "equipment_maintenance.pdf"
    },
    {
        "question": "What is the transformer maintenance schedule?",
        "ground_truth": "Transformer maintenance schedule: Oil level check weekly (Junior Engineer), oil temperature check daily (Operator), oil dielectric strength test every 6 months (Senior Engineer), Buchholz relay test annually (Certified Technician), full shutdown maintenance as per manufacturer schedule.",
        "source": "equipment_maintenance.pdf"
    },

    # ── OUTAGE PROCEDURES QUESTIONS ──────────────────────────
    {
        "question": "How much notice must be given to consumers before a planned outage?",
        "ground_truth": "All planned outages must be intimated to affected consumers minimum 24 hours in advance through SMS, local newspapers and community notice boards.",
        "source": "outage_procedures.pdf"
    },
    {
        "question": "How far in advance must a planned outage request be submitted?",
        "ground_truth": "Planned outage request must be submitted to load despatch center minimum 48 hours before the outage date with full details of work scope.",
        "source": "outage_procedures.pdf"
    },
    {
        "question": "What are the permitted hours for planned outages?",
        "ground_truth": "Planned outages must be scheduled between 10 AM and 4 PM on working days.",
        "source": "outage_procedures.pdf"
    },
]

if __name__ == "__main__":
    print(f"Total test questions: {len(TEST_DATA)}")
    for i, item in enumerate(TEST_DATA, 1):
        print(f"\n{i}. [{item['source']}]")
        print(f"   Q: {item['question']}")
        print(f"   A: {item['ground_truth'][:80]}...")