# vaccine_data.py
"""
Small helper to prepare and index the vaccine dataset into VectorDB.
"""
from typing import Any

def prepare_and_index(vector_db):
    """
    Index the standard vaccine schedule (simple textual descriptions).
    vector_db: instance of VectorDB (from db_helpers)
    """
    STANDARD_VACCINES = [
        {"name": "BCG", "due_month": 0, "desc": "BCG vaccine at birth; prevents severe forms of TB."},
        {"name": "OPV (birth)", "due_month": 0, "desc": "Oral polio vaccine at birth."},
        {"name": "Hepatitis B (birth)", "due_month": 0, "desc": "First dose of Hepatitis B at birth."},
        {"name": "OPV (6 weeks)", "due_month": 1.5, "desc": "OPV follow-up at 6 weeks."},
        {"name": "DPT (6 weeks)", "due_month": 1.5, "desc": "DPT at 6 weeks; protects diphtheria, pertussis and tetanus."},
        {"name": "Measles (9 months)", "due_month": 9, "desc": "Measles vaccine around 9 months in many schedules."},
        {"name": "MMR (15-18 months)", "due_month": 15, "desc": "MMR given between 15-18 months."}
    ]
    for v in STANDARD_VACCINES:
        text = f"{v['name']} — due at approximately {v['due_month']} months. {v['desc']}"
        vector_db.add(text, meta={"name": v["name"], "due_month": v["due_month"]})
    return True
