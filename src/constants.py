# src/constants.py
"""
After exploring the dataset, these features where chosen
(not all, these dicts mostly cover most difficult and long csvs available: lab and chart events)
Project-wide constants:
- final itemids for v0 features (manually curated)
- documented lab itemids chosen by our selection logic (v2)

"""

# -----------------------------
# Vitals (chartevents) - v0
# -----------------------------
VITAL_ITEMIDS = {
    # Routine vital signs
    "hr": [220045],        # Heart Rate
    "rr": [220210],        # Respiratory Rate
    "spo2": [220277],      # O2 saturation pulseoxymetry
    "temp_c": [223762],    # Temperature Celsius
    # Non-invasive BP
    "nibp_sys": [220179],  # Non Invasive Blood Pressure systolic
    "nibp_dia": [220180],  # Non Invasive Blood Pressure diastolic
    "nibp_mean": [220181], # Non Invasive Blood Pressure mean
}

# Optional (v0.1 / v1 candidates) - documented, not used in v0 by default
OPTIONAL_VITAL_ITEMIDS = {
    "o2_flow": [223834],   # O2 Flow (respiratory support proxy)
}

# -----------------------------
# Labs (labevents) - documented v0 canonical choices
# NOTE: these are the itemids your v2 script selected as most frequent
# in the 6h window for the cohort.
# -----------------------------
LAB_ITEMIDS_V0_DOC = {
    "bicarbonate": 50882,  # Bicarbonate
    "creatinine": 50912,   # Creatinine
    "hemoglobin": 51222,   # Hemoglobin
    "lactate": 50813,      # Lactate
    "platelets": 51265,    # Platelet Count
    "potassium": 50971,    # Potassium
    "sodium": 50983,       # Sodium
    "wbc": 51516,          # WBC
}

# HbA1c itemid (baseline historical covariate)
HBA1C_ITEMID = 50852

# General / anthropometrics from chartevents
GENERAL_ITEMIDS = {
    "admission_weight_kg": [226512],  # Admission Weight (Kg)
    "height_cm": [226730],            # Height (cm)
}

