import json
import re
import uuid
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm  # pip install tqdm

BASE_URL = "http://localhost:1234/v1"


def pick_first_model(base_url: str) -> str:
    r = requests.get(f"{base_url}/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


MODEL = pick_first_model(BASE_URL)
print("Using model:", MODEL)

ROOT = Path(__file__).resolve().parents[3]
dataset_card_path = ROOT / "sepsis_dataset_card.json"
csv_path = ROOT / "sepsis.csv"

out_json_path = ROOT / "cate_llm_results.json"
out_csv_path = ROOT / "sepsis_with_cate_llm.csv"

TREATMENT = "treat_early"
OUTCOME = "y_hosp_mort"

LEAKY_COLS = {
    "subject_id", "hadm_id", "stay_id",
    "intime", "sepsis_time", "t0_time", "exposure_end",
    "first_diur_time", "dischtime", "deathtime",
}

# Budget conservativo per restare sotto ~4096 token
MAX_PROMPT_CHARS = 12000


def load_dataset_card(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_cate(text: str) -> Optional[float]:
    s = text.strip()
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", s):
        v = float(s)
        return v if -1.0 <= v <= 1.0 else None
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", s)
    if len(nums) == 1:
        v = float(nums[0])
        return v if -1.0 <= v <= 1.0 else None
    return None


def call_lmstudio_stateless(user_content: str, temperature: float = 0.0, timeout: int = 1800) -> str:
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": 16,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]


def get_all_variable_names_from_card(card: Dict[str, Any]) -> List[str]:
    cols = card.get("columns", {})
    if not isinstance(cols, dict) or len(cols) == 0:
        raise KeyError("Dataset card missing 'columns' dict with per-variable stats.")
    return list(cols.keys())


def build_X_from_card(card: Dict[str, Any]) -> List[str]:
    all_vars = get_all_variable_names_from_card(card)
    X = [v for v in all_vars if v not in LEAKY_COLS and v not in {TREATMENT, OUTCOME}]
    return X


def _round_num(x: Any, nd: int = 3) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    return x


def compact_variable_desc(vdesc: Dict[str, Any]) -> Dict[str, Any]:
    """
    SUPER-compatto per stare nei 4096 token:
    - dtype, role, missing_rate (arrotondata)
    - numeriche: mean, p50, p95 (arrotondate)
    - categoriche: top 3 (pct arrotondate)
    """
    out: Dict[str, Any] = {}
    for k in ["dtype", "role", "missing_rate"]:
        if k in vdesc:
            out[k] = _round_num(vdesc[k], 4)

    if "top_values" in vdesc and isinstance(vdesc["top_values"], list):
        tv = vdesc["top_values"][:3]
        for item in tv:
            if isinstance(item, dict) and "pct" in item:
                item["pct"] = _round_num(item["pct"], 4)
        out["top_values"] = tv

    stats = vdesc.get("stats")
    if isinstance(stats, dict):
        keep_stats = {}
        for sk in ["mean", "p50", "p95"]:
            if sk in stats:
                keep_stats[sk] = _round_num(stats[sk], 3)
        if keep_stats:
            out["stats"] = keep_stats

    return out


def build_global_dataset_meta(card: Dict[str, Any], X: List[str]) -> Dict[str, Any]:
    """
    Meta globale leggero: NIENTE lista X_names (costosa).
    """
    cols = card["columns"]
    meta: Dict[str, Any] = {
        "dataset_name": card.get("dataset_name"),
        "n_rows": card.get("n_rows"),
        "n_cols": card.get("n_cols"),
        "treatment": TREATMENT,
        "outcome": OUTCOME,
        "n_covariates": len(X),
    }
    for special in [TREATMENT, OUTCOME]:
        if special in cols and isinstance(cols[special], dict):
            meta[special] = compact_variable_desc(cols[special])
    return meta


def build_patient_conditioned_dataset_desc(
    card: Dict[str, Any],
    global_meta: Dict[str, Any],
    patient_vars: List[str],
) -> Dict[str, Any]:
    cols = card["columns"]
    detailed: Dict[str, Any] = {}
    for v in patient_vars:
        vdesc = cols.get(v)
        if isinstance(vdesc, dict):
            detailed[v] = compact_variable_desc(vdesc)
    return {"global": global_meta, "detailed_for_patient_vars": detailed}


def row_to_patient_dict(row: pd.Series) -> Dict[str, Any]:
    d = row.to_dict()
    for k, v in list(d.items()):
        if pd.isna(v):
            d[k] = None
    return d


def filter_patient(patient: Dict[str, Any], X: List[str]) -> Dict[str, Any]:
    keep = set(X) | {TREATMENT, OUTCOME}
    return {k: patient.get(k, None) for k in keep}


def compact_patient_payload(patient_slim: Dict[str, Any], X: List[str]) -> Dict[str, Any]:
    """
    Evita 'var: null' per 40 colonne: token killer.
    NON rimuove covariate: le mette in missing_covariates.
    """
    observed: Dict[str, Any] = {}
    missing: List[str] = []

    for k in X:
        v = patient_slim.get(k, None)
        if v is None:
            missing.append(k)
        else:
            observed[k] = v

    # assicurati di includere has_* sempre, se presenti
    for k, v in patient_slim.items():
        if k.startswith("has_"):
            observed[k] = v

    return {
        "treatment": patient_slim.get(TREATMENT, None),
        "outcome": patient_slim.get(OUTCOME, None),
        "observed_covariates": observed,
        "missing_covariates": missing,
    }


def build_user_message(dataset_desc: Dict[str, Any], patient_payload: Dict[str, Any], nonce: str) -> str:
    return (
        f"REQUEST_NONCE={nonce}\n"
        "DATASET_DESCRIPTION_JSON:\n"
        f"{json.dumps(dataset_desc, ensure_ascii=False)}\n\n"
        "PATIENT_INSTANCE_JSON:\n"
        f"{json.dumps(patient_payload, ensure_ascii=False)}\n\n"
        "Return ONLY the CATE as a single number in [-1.0, 1.0]."
    )


def build_user_message_with_budget(
    card: Dict[str, Any],
    global_meta: Dict[str, Any],
    patient_payload: Dict[str, Any],
) -> Tuple[str, str, int]:
    """
    Riduce progressivamente quante variabili includere in detailed_for_patient_vars
    (ma il paziente completo è sempre rappresentato con observed+missing).
    """
    observed_keys = list(patient_payload.get("observed_covariates", {}).keys())
    observed_keys.sort()

    has_vars = [k for k in observed_keys if k.startswith("has_")]
    other_vars = [k for k in observed_keys if not k.startswith("has_")]

    k = len(other_vars)

    while True:
        used_vars = has_vars + other_vars[:k]
        dataset_desc = build_patient_conditioned_dataset_desc(card, global_meta, used_vars)

        nonce = uuid.uuid4().hex
        user_msg = build_user_message(dataset_desc, patient_payload, nonce=nonce)

        if len(user_msg) <= MAX_PROMPT_CHARS:
            return user_msg, nonce, len(used_vars)

        # se non rientra, riduci ancora
        if k <= 5:
            k = 0  # fallback: niente detailed (solo global)
        else:
            k = max(5, k // 2)


def estimate_cate_for_all_rows(
    dataset_card_path: str,
    csv_path: str,
    limit: Optional[int] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:

    card = load_dataset_card(dataset_card_path)
    df = load_csv(csv_path)
    if limit is not None:
        df = df.iloc[:limit].copy()

    X = build_X_from_card(card)
    global_meta = build_global_dataset_meta(card, X)

    results_json: List[Dict[str, Any]] = []
    cates: List[Optional[float]] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Estimating CATE"):
        patient = row_to_patient_dict(row)
        patient_slim = filter_patient(patient, X)

        patient_payload = compact_patient_payload(patient_slim, X)

        user_msg, nonce, n_used = build_user_message_with_budget(card, global_meta, patient_payload)

        out = call_lmstudio_stateless(user_msg, temperature=0.0)
        cate = parse_cate(out)

        if cate is None:
            out2 = call_lmstudio_stateless(
                user_msg + "\nABSOLUTE OUTPUT RULE: output ONLY one number.",
                temperature=0.0
            )
            cate = parse_cate(out2)

        cates.append(cate)
        results_json.append({
            "row_idx": int(idx),
            "cate_llm": cate,
            TREATMENT: patient.get(TREATMENT, None),
            OUTCOME: patient.get(OUTCOME, None),
            "nonce": nonce,
            "n_detailed_vars_used": n_used,
            "prompt_chars": len(user_msg),
            "n_observed_covariates": len(patient_payload.get("observed_covariates", {})),
            "n_missing_covariates": len(patient_payload.get("missing_covariates", [])),
        })

    df["cate_llm"] = cates
    return df, results_json


if __name__ == "__main__":
    df_out, results = estimate_cate_for_all_rows(
        str(dataset_card_path),
        str(csv_path),
        limit=None
    )

    df_out.to_csv(out_csv_path, index=False)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved CSV:", out_csv_path)
    print("Saved JSON:", out_json_path)