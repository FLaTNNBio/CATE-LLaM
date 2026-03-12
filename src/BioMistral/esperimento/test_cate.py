import os
import json
import pandas as pd
import time
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.config import ANALYTIC_DIR

DATASET_PATH = ANALYTIC_DIR / "analytic_sepsis_early_diuretics_v1.parquet"
PROMPT_PATH = Path("src") / "BioMistral" / "prompts" / "prompt.txt"
OUTPUT_FILE = Path("result_cate_llm") / "calculated_cate_llama.csv"

CHUNK_SIZE = 8
MAX_RETRIES = 3


def strict_parse_array(text):
    text = text.strip()

    if not text.startswith("[") or not text.endswith("]"):
        return None

    try:
        data = json.loads(text)
        return data
    except:
        return None
def validate_output(parsed, expected_indices):

    if parsed is None:
        return False, "Parsing failed"

    if not isinstance(parsed, list):
        return False, "Output is not a list"

    if len(parsed) != len(expected_indices):
        return False, "Length mismatch"

    for item in parsed:

        if not isinstance(item, dict):
            return False, "Element not dict"

        if "cate" not in item:
            return False, "Missing cate"

        if not isinstance(item["cate"], (int, float)):
            return False, "Cate not numeric"

        if not (-1.0 <= float(item["cate"]) <= 1.0):
            return False, "Out of bounds"

    return True, "OK"


def get_cate_estimates():
    client = InferenceClient(api_key=hf_token)
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    try:
        df = pd.read_parquet(DATASET_PATH)
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            base_prompt = f.read()
    except Exception as e:
        print(f"❌ Errore caricamento file: {e}")
        return

    print(f"📊 Dataset caricato: {len(df)} righe.")
    print(f"🤖 Modello: {model_id}")
    print(f"📦 Chunk size: {CHUNK_SIZE}")

    # Crea cartella output se non esiste
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Se file non esiste scrivi header
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        processed_indices = set(existing["original_index"])
    else:
        processed_indices = set()
        pd.DataFrame(columns=["original_index", "cate_estimate"]).to_csv(OUTPUT_FILE, index=False)

    # Ciclo a blocchi da 8
    for start_idx in range(0, len(df), CHUNK_SIZE):

        chunk = df.iloc[start_idx:start_idx + CHUNK_SIZE]
        chunk = chunk[~chunk.index.isin(processed_indices)]
        print(f"\n🔄 Elaborazione righe {start_idx} - {start_idx + len(chunk) - 1}")

        # Costruzione prompt con 8 righe
        if chunk.empty:
            continue

        expected_indices = list(chunk.index)

        chunk_text = ""
        for idx, row in chunk.iterrows():
            chunk_text += f"\nPatient index: {idx}\n"
            chunk_text += row.to_string()
            chunk_text += "\n---\n"

        full_prompt = base_prompt.replace("{row_data}", chunk_text)
        full_prompt += f"\nYou must output exactly {len(expected_indices)} results."

        attempts = 0
        success = False
        while attempts < MAX_RETRIES and not success:

            response = client.chat_completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": "Deterministic clinical engine."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=800,
                temperature=0.0,
            )

            raw_output = response.choices[0].message.content.strip()

            parsed = strict_parse_array(raw_output)
            valid, reason = validate_output(parsed, expected_indices)

            if valid:
                rows_to_save = []
                for idx, item in zip(expected_indices, parsed):
                    rows_to_save.append({
                        "original_index": idx,
                        "cate_estimate": float(item["cate"])
                    })

                pd.DataFrame(rows_to_save) \
                    .to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

                processed_indices.update(expected_indices)
                success = True
                print(f"✅ Salvati {len(rows_to_save)} pazienti")

            else:
                print(f"⚠️ Tentativo {attempts + 1} fallito: {reason}")
                attempts += 1
                time.sleep(2)

        if not success:
            print(f"❌ Chunk {start_idx} fallito dopo {MAX_RETRIES} tentativi.")

    print(f"\n🎯 Elaborazione completata. File salvato in {OUTPUT_FILE}")


if __name__ == "__main__":

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("❌ HF_TOKEN non trovato.")
        os.abort()

    get_cate_estimates()