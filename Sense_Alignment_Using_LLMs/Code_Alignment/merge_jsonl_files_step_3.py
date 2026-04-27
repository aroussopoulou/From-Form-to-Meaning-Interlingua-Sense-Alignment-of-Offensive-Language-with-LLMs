import json
import re
from pathlib import Path
import pandas as pd
from collections import defaultdict

# PATHS
IN_DIR  = Path(r"PATH\alignments_files\files_jsonl\Batches\arit")
CSV_PATH = Path(r"PATH\Alignment_File\hurtnet_total_dataset.csv")
OUT_DIR = Path(r"PATH\alignments_files\files_jsonl\Total")
OUT_FILE = OUT_DIR / "alignment_arit.jsonl"

# bgit_{row1}_{row2} 
ID_RE = re.compile(r"^arit_(\d+)_(\d+)$")

def normalize_null(x):
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"null", "none", "nan", ""}:
        return None
    return x

def parse_model_content(content: str) -> dict:
    if content is None:
        return {}
    s = str(content).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
    except Exception:
        return {"raw_content": s}

    if isinstance(obj, dict):
        if "confidence" in obj:
            try:
                obj["confidence"] = float(obj["confidence"])
            except Exception:
                pass
        for k in ("globalsense_id", "globalsense_label_en"):
            if k in obj:
                obj[k] = normalize_null(obj[k])
        if "decision" in obj and isinstance(obj["decision"], str):
            obj["decision"] = obj["decision"].strip().lower()
    return obj if isinstance(obj, dict) else {"parsed": obj}

def extract_message_content(line_obj: dict):
    resp = line_obj.get("response") or {}
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        return msg.get("content")
    return line_obj.get("content")

def load_rowid_to_lemma(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if "Row_id" not in df.columns or "Lemma" not in df.columns:
        raise ValueError("CSV must contain columns: Row_id, Lemma")
    return dict(zip(df["Row_id"].astype(int), df["Lemma"].astype(str)))

def find_jsonl_files(root: Path):
    # recursive + both cases
    files = list(root.rglob("*.jsonl")) + list(root.rglob("*.JSONL"))
    # remove duplicates 
    uniq = sorted(set(files))
    return uniq

def main():
    if not IN_DIR.exists():
        raise SystemExit(f"Input folder not found: {IN_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rowid2lemma = load_rowid_to_lemma(CSV_PATH)

    jsonl_files = find_jsonl_files(IN_DIR)
    if not jsonl_files:
        raise SystemExit(f"No .jsonl files found under: {IN_DIR}")

    # diagnostics per file
    per_file = defaultdict(lambda: defaultdict(int))
    examples_bad_id = defaultdict(list)

    written_total = 0

    with OUT_FILE.open("w", encoding="utf-8") as fout:
        for fp in jsonl_files:
            with fp.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue

                    per_file[fp]["lines"] += 1

                    try:
                        obj = json.loads(line)
                    except Exception:
                        per_file[fp]["bad_json"] += 1
                        continue

                    custom_id = obj.get("custom_id") or obj.get("alignment_id")
                    if not custom_id:
                        per_file[fp]["no_custom_id"] += 1
                        continue

                    m = ID_RE.match(custom_id)
                    if not m:
                        per_file[fp]["id_mismatch"] += 1
                        if len(examples_bad_id[fp]) < 5:
                            examples_bad_id[fp].append(custom_id)
                        continue

                    row1 = int(m.group(1))
                    row2 = int(m.group(2))

                    it_row = row1
                    ar_row = row2

                    ar_lemma = rowid2lemma.get(ar_row)
                    it_lemma = rowid2lemma.get(it_row)

                    content = extract_message_content(obj)
                    parsed = parse_model_content(content)

                    out = {
                        "alignment_id": custom_id,
                        "ar_row_id": ar_row,
                        "it_row_id": it_row,
                        "ar_lemma": ar_lemma,
                        "it_lemma": it_lemma,
                        "response": {
                            "decision": parsed.get("decision"),
                            "confidence": parsed.get("confidence"),
                            "globalsense_id": parsed.get("globalsense_id"),
                            "globalsense_label_en": parsed.get("globalsense_label_en"),
                            "rationale_en": parsed.get("rationale_en"),
                        }
                    }

                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    per_file[fp]["written"] += 1
                    written_total += 1

    # REPORT
    print(f"\nFound {len(jsonl_files)} jsonl files under: {IN_DIR}")
    print(f"Output: {OUT_FILE}")
    print(f"TOTAL written lines: {written_total}\n")

    zero_written = []
    for fp in jsonl_files:
        stats = per_file[fp]
        if stats["written"] == 0:
            zero_written.append(fp)

        print(f"FILE: {fp.name}")
        print(f"  lines:       {stats['lines']}")
        print(f"  written:     {stats['written']}")
        print(f"  bad_json:    {stats['bad_json']}")
        print(f"  no_custom_id:{stats['no_custom_id']}")
        print(f"  id_mismatch: {stats['id_mismatch']}")
        if examples_bad_id[fp]:
            print(f"  examples of custom_id mismatch: {examples_bad_id[fp]}")
        print()

    if zero_written:
        print("FILES WITH 0 WRITTEN LINES (likely your 'missing' batch):")
        for fp in zero_written:
            print(" ", fp)

if __name__ == "__main__":
    main()
