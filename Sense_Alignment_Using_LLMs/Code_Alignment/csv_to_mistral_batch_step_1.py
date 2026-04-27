import os, json, tqdm
from pathlib import Path
import pandas as pd

"""
Δημιουργεί .jsonl batch-files από το hurtnet_total_dataset.csv
Έτοιμα για upload στο Batch API τoυ Mistral.
"""

# Parameters 
CSV_PATH = Path(r"PATH\Alignment_File\hurtnet_total_dataset.csv")    
BATCH_DIR = Path(r"PATH\alignments_files\files_jsonl\First_Batches\batches_el_fr")                   
BATCH_SIZE = 10_000                                
MODEL      = "mistral-small-latest"
SYSTEM_PROMPT = """
You are an expert in offensive language and multilingual lexicography.

Your task: given TWO OFFENSIVE TERMS (Term A and Term B), decide how similar they are in sense of their offensive meaning and use.

You must choose exactly ONE of:
- merge   → same offensive sense (one Sense)
- related → clearly related / near-synonyms, but not exactly the same sense
- unrelated → no useful offensive relation

Guidelines (short):
- Use "merge" only if they share:
  (a) same offensive purpose (e.g. expletive for a bad event, insult for the same type of person),
  (b) same typical target (situation/person/group),
  (c) same core offensive idea and similar strength.
- Use "related" if they are in the same offensive area but differ in purpose, target or strength.
- Use "unrelated" if their offensive domain or purpose is different, or any link is too vague.

Be conservative:
- If between merge/related → choose related.
- If between related/unrelated → choose unrelated.

OUTPUT FORMAT (VERY IMPORTANT):
Answer with EXACTLY ONE LINE, with 5 fields separated by "|||", in this order:

decision|||confidence|||globalsense_id|||globalsense_label_en|||rationale_en

Where:
- decision: one of merge, related, unrelated (lowercase)
- confidence: a number between 0 and 1 with at most 2 decimals (e.g. 0.87)
- sense_id:
    - if decision is merge or related: short snake_case id, e.g. gs_insult_stupid_person
    - if decision is unrelated: write null
- sense_label_en:
    - if decision is merge or related: short English label (max 12 words)
    - if decision is unrelated: write null
- rationale_en: 1–2 short sentences in English explaining your decision
  (mention purpose, target, and core offensive idea; do NOT print any slurs)

Your entire reply MUST be this single line.
Do NOT add explanations before or after.
Do NOT use markdown or code fences.
""".strip()

# Upload the csv and split it for Bulgarian and Greek and prepare the folder
df   = pd.read_csv(CSV_PATH, encoding="utf-8")
el  = df[df.Language == "Greek"].reset_index(drop=True)
fr = df[df.Language == "French"].reset_index(drop=True)

print("Greek entries:", len(el))
print("French entries:", len(fr))

BATCH_DIR.mkdir(exist_ok=True)

# Write the batch size files
file_id, line_id = 0, 0
f = open(BATCH_DIR / f"batch_{file_id}.jsonl", "w", encoding="utf-8")

for g in tqdm.tqdm(fr.itertuples(), total=len(fr), desc="French lemmas"):
    for b in el.itertuples():
        ridA = int(g.Row_id) if pd.notna(g.Row_id) else g.Index + 1
        ridB = int(b.Row_id) if pd.notna(b.Row_id) else b.Index + 1
        custom_id = f"elfr_{ridA}_{ridB}"          

        user_msg = (
            f"Sense A (Greek):\n"
            f"  lemma: {g.Lemma}\n"
            f"  definition: {g.Definition or ''}\n"
            f"  example: {g.Example or ''}\n\n"
            f"Sense B (French):\n"
            f"  lemma: {b.Lemma}\n"
            f"  definition: {b.Definition or ''}\n"
            f"  example: {b.Example or ''}"
        )

        record = {
            "custom_id": custom_id,
            "body": {
                "model": MODEL,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg}
                ]
            }
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        line_id += 1

        if line_id % BATCH_SIZE == 0:
            f.close()
            file_id += 1
            f = open(BATCH_DIR / f"batch_{file_id}.jsonl", "w", encoding="utf-8")

f.close()
print(f"✓ Ολοκληρώθηκε – γράφτηκαν {line_id:,} ζεύγη σε {file_id+1} αρχεία των 10 000 γραμμών")