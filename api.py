# ============================================================
# api.py
# Shared FastAPI server for both donor-matching and disease-prediction.
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Test UI:  http://localhost:8000/docs
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import sys

load_dotenv()

# Feature folders — resolved from this file's own location so it works
# no matter what directory the server is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DONOR_DIR = os.path.join(BASE_DIR, "donor-matching")
DISEASE_DIR = os.path.join(BASE_DIR, "disease-prediction")

# donor-matching's modules use flat (non-package) imports internally,
# so they need their own folder on sys.path to resolve.
sys.path.insert(0, DONOR_DIR)

from top5_matcher import find_top5_donors
import requests
import pickle

app = FastAPI(
    title="Bone Marrow Donor Matching API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for disease prediction
DISEASE_MODEL_PATH = os.path.join(DISEASE_DIR, "model.pkl")
DISEASE_ENCODER_PATH = os.path.join(DISEASE_DIR, "encoder.pkl")
SYMPTOMS_PATH = os.path.join(DISEASE_DIR, "symptoms.pkl")

disease_model = pickle.load(open(DISEASE_MODEL_PATH, "rb"))
disease_encoder = pickle.load(open(DISEASE_ENCODER_PATH, "rb"))
SYMPTOM_NAMES = pickle.load(open(SYMPTOMS_PATH, "rb"))

SUPABASE_URL, SUPABASE_KEY = (
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)

# SUPABASE_URL is just the project base (e.g. https://xxx.supabase.co) —
# every table lives under /rest/v1/<table>.
SUPABASE_REST_URL = f"{SUPABASE_URL}/rest/v1"

supabase_headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

# ============================================================
# Disease / disease-group values coming from the frontend don't
# always match the exact casing the models were trained on
# (e.g. "Chronic", "Non-malignant"). Normalize them here so the
# ML pipeline sees the vocabulary it actually knows.
# ============================================================

DISEASE_TYPE_MAP = {
    "leukemia":     "AML",
    "aml":          "AML",
    "all":          "ALL",
    "chronic":      "chronic",
    "lymphoma":     "lymphoma",
    "nonmalignant": "nonmalignant",
}

def normalize_disease_type(value):
    key = str(value).strip().lower().replace("-", "").replace(" ", "")
    return DISEASE_TYPE_MAP.get(key, "AML")

def normalize_disease_group(value):
    key = str(value).strip().lower().replace("-", "").replace(" ", "")
    return key if key in ("malignant", "nonmalignant") else "malignant"

def normalize_cmv(value):
    key = str(value).strip().lower()
    if key in ("present", "positive", "pos", "+", "yes"):
        return "present"
    return "absent"

#disease prediction
class DiseaseRequest(BaseModel):
    symptoms: list[str]

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"message": "Bone Marrow Matching API is running ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/find-top5/{patient_id}")
def find_top5(patient_id: int):

    # Fetch patient
    patient_res = requests.get(
        f"{SUPABASE_REST_URL}/Patient",
        headers=supabase_headers,
        params={"Patient_id": f"eq.{patient_id}"}  # capital P
    )

    patient_data = patient_res.json()
    print("Patient data:", patient_data)
    
    if not patient_data:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = patient_data[0]

    # Rename + map patient fields to match with upa table
    patient = {
        "patient_id": patient["Patient_id"],

        "recipient_age": float(patient["Age"]),
        "recipient_gender": patient["Gender"].lower(),
        "recipient_body_mass": float(patient["BodyMass"]),
        "recipient_ABO": patient["BloodGroup"].replace("O", "0"),
        "recipient_rh": {
            "+": "plus", "-": "minus",
            "positive": "plus", "negative": "minus",
            "plus": "plus", "minus": "minus"          # ← add these
        }.get(patient["RhFactor"].lower(), "plus"),

        "recipient_CMV": normalize_cmv(patient["CMVStatus"]),
        "disease": normalize_disease_type(patient["DiseaseType"]),
        "disease_group": normalize_disease_group(patient["DiseaseGroup"]),
        "risk_group": str(patient["RiskGroup"]).strip().lower(),
        "tx_post_relapse": str(patient["PostRelapse"]).strip().lower(),

        # HLA fields 
        "HLA_A_1": patient["Hla_a_1"],
        "HLA_A_2": patient["Hla_a_2"],
        "HLA_B_1": patient["Hla_b_1"],
        "HLA_B_2": patient["Hla_b_2"],
        "HLA_C_1": patient["Hla_c_1"],
        "HLA_C_2": patient["Hla_c_2"],
        "HLA_DRB1_1": patient["Hla_drb1_1"],
        "HLA_DRB1_2": patient["Hla_drb1_2"],
        "HLA_DQB1_1": patient["Hla_dqb1_1"],
        "HLA_DQB1_2": patient["Hla_dqb1_2"],
    }

    # Fetch all donors
    donor_res = requests.get(
        f"{SUPABASE_REST_URL}/Donor",
        headers=supabase_headers
    )

    donors = donor_res.json()

    if not donors:
        raise HTTPException(status_code=404, detail="No donors found")

    donor_df = pd.DataFrame(donors)

    # Rename columns for donor to match with table
    donor_df = donor_df.rename(columns={
        "Donor_id": "donor_id",
        "Age": "donor_age",
        "BloodGroup": "donor_ABO",
        "CMVStatus": "donor_CMV",
        "Gender": "donor_gender",
        "StemCellSource": "stem_cell_source",

        "Hla_a_1": "HLA_A_1",
        "Hla_a_2": "HLA_A_2",
        "Hla_b_1": "HLA_B_1",
        "Hla_b_2": "HLA_B_2",
        "Hla_c_1": "HLA_C_1",
        "Hla_c_2": "HLA_C_2",
        "Hla_drb1_1": "HLA_DRB1_1",
        "Hla_drb1_2": "HLA_DRB1_2",
        "Hla_dqb1_1": "HLA_DQB1_1",
        "Hla_dqb1_2": "HLA_DQB1_2",
    })

    # Add missing fields
    donor_df["CD34_x1e6_per_kg"] = 10.0
    donor_df["CD3_x1e8_per_kg"] = 5.0
    donor_df["CD3_to_CD34_ratio"] = donor_df["CD3_x1e8_per_kg"] / donor_df["CD34_x1e6_per_kg"] 

    # Fix formats
    donor_df["donor_ABO"] = donor_df["donor_ABO"].replace({"O": "0"})
    donor_df["donor_CMV"] = donor_df["donor_CMV"].map(normalize_cmv)

    # Run ML
    top5 = find_top5_donors(patient, donor_df)

    # Prepare insert data
    insert_data = []

    for rank, d in enumerate(top5, 1):
        insert_data.append({
            "Patient_id": int(patient["patient_id"]),
            "Patient_Name": f"Patient {patient['patient_id']}",

            "Donor_id": d["donor_id"],
            "Donor_Name": f"Donor {d['donor_id']}",

            "CompatabilityScore": d["compatibility_score"],
            "HlaMatch": d["hla_match"],
            "AboMatch": d["abo_match"],

            "Antigen":d["antigen_diff"],
            "Allele":d["allel_diff"],

            "Survival": d["alive_probability"],
            "RelapseRisk": d["relapse_risk"],
            "GvhdRisk": d["gvhd_risk"]
        })

    # Delete existing matches for this patient before inserting new ones
    requests.delete(
        f"{SUPABASE_REST_URL}/Patient-Donor",
        headers=supabase_headers,
        params={"Patient_id": f"eq.{patient['patient_id']}"}
    )

    #  Insert into Supabase
    res = requests.post(
        f"{SUPABASE_REST_URL}/Patient-Donor",
        json=insert_data,
        headers=supabase_headers
    )

    if res.status_code >= 300:
        raise HTTPException(status_code=500, detail=res.text)

    return {
        "message": "Top 5 generated and stored ✅",
        "data": insert_data
    }

@app.get("/symptoms")
def get_symptoms():
    return {"symptoms": sorted(SYMPTOM_NAMES)}

@app.post("/predict-disease")
def predict_disease(request: DiseaseRequest):
    try:
        symptoms_list = request.symptoms

        input_vec = np.zeros(len(SYMPTOM_NAMES), dtype=np.float32)
        unrecognised = []

        for sym in symptoms_list:
            sym_lower = sym.strip().lower()
            matched = [i for i, s in enumerate(SYMPTOM_NAMES) if s.lower() == sym_lower]

            if matched:
                input_vec[matched[0]] = 1
            else:
                unrecognised.append(sym)

        probabilities = disease_model.predict_proba([input_vec])[0]
        top_indices = np.argsort(probabilities)[::-1][:5]

        results = []
        for idx in top_indices:
            results.append({
                "disease": disease_encoder.inverse_transform([idx])[0],
                "confidence": round(float(probabilities[idx]) * 100, 2)
            })

        return {
            "predictions": results,
            "unrecognised_symptoms": unrecognised,
            "low_confidence": bool(probabilities[top_indices[0]] < 0.30)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))