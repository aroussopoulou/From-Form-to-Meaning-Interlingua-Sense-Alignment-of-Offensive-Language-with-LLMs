import json
from pathlib import Path
import pandas as pd
from itertools import product

# CONFIG
LANGS = ["ar", "bg", "el", "fr", "it"]
ALL_PAIRS_5 = [
    ("ar","bg"),("ar","el"),("ar","fr"),("ar","it"),
    ("bg","el"),("bg","fr"),("bg","it"),
    ("el","fr"),("el","it"),
    ("fr","it")
]

PIVOT_LANG = "it"
CAP_PER_LANG = 25
MAX_CHECKS_PER_PIVOT = 20000
WRITE_AUDIT = True

# Near-misses:
AUDIT_ONLY_BUCKETS = {"9/10", "8/10"} 

# For checking the disk space
MAX_ROWS_PER_BUCKET = None 

# CSV encoding
CSV_ENCODING = "utf-8-sig"

# HELPERS
def pair_from_alignment_id(alignment_id: str):
    prefix = alignment_id.split("_")[0]
    return prefix[:2], prefix[2:4]

def node(lang, row_id):
    return f"{lang}:{row_id}"

def keypair(n1, n2):
    return frozenset([n1, n2])

def pairs_ok_bucket(missing, nonmerge):
    pairs_ok = 10 - len(missing) - len(nonmerge)
    return pairs_ok, f"{pairs_ok}/10"

def safe_row_id(n):
    return n.split(":", 1)[1] if ":" in n else n


# Build indexes
def build_indexes(jsonl_dir: str):
    decision_index = {}
    merge_neighbors = {}

    jsonl_paths = sorted(Path(jsonl_dir).glob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No .jsonl found in {jsonl_dir}")

    for p in jsonl_paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                d = json.loads(line)
                resp = d.get("response", {}) or {}
                decision = (resp.get("decision") or "").strip().lower()
                if decision not in {"merge","related","unrelated"}:
                    continue

                alignment_id = d.get("alignment_id")
                if not alignment_id:
                    continue

                la, lb = pair_from_alignment_id(alignment_id)
                ra = d.get(f"{la}_row_id")
                rb = d.get(f"{lb}_row_id")
                if ra is None or rb is None:
                    continue

                na = node(la, str(ra))
                nb = node(lb, str(rb))
                k = keypair(na, nb)

                rec = {
                    "decision": decision,
                    "confidence": float(resp.get("confidence", 0.0) or 0.0),
                    "rationale_en": resp.get("rationale_en", None),
                    "alignment_id": alignment_id
                }

                if k not in decision_index:
                    decision_index[k] = rec
                else:
                    old = decision_index[k]
                    if old["decision"] != "merge" and rec["decision"] == "merge":
                        decision_index[k] = rec

                if decision == "merge":
                    for src, dst, dst_lang in [(na, nb, lb), (nb, na, la)]:
                        merge_neighbors.setdefault(src, {}).setdefault(dst_lang, set()).add(dst)

    return decision_index, merge_neighbors

# Verify 5-tuple
def verify_5tuple(dec_index, nodes_by_lang):
    missing = []
    nonmerge = []

    for (a,b) in ALL_PAIRS_5:
        n1 = nodes_by_lang[a]
        n2 = nodes_by_lang[b]
        rec = dec_index.get(keypair(n1, n2))

        if rec is None:
            missing.append(f"{a}-{b}")
        else:
            if rec["decision"] != "merge":
                nonmerge.append({
                    "pair": f"{a}-{b}",
                    "decision": rec["decision"],
                    "confidence": rec["confidence"],
                    "alignment_id": rec["alignment_id"],
                    "rationale_en": rec["rationale_en"]
                })

    ok = (len(missing) == 0 and len(nonmerge) == 0)
    return ok, missing, nonmerge


# Streaming CSV writer per bucket
class BucketWriters:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.files = {}          # bucket -> filepath
        self.header_written = set()
        self.counts = {}         # bucket -> rows written

    def _path_for_bucket(self, bucket: str):
        # "9/10" -> "audit_failed_9of10.csv"
        name = bucket.replace("/", "of")
        return self.out_dir / f"audit_failed_{name}.csv"

    def write_row(self, bucket: str, row: dict):
        if not WRITE_AUDIT:
            return

        if AUDIT_ONLY_BUCKETS is not None and bucket not in AUDIT_ONLY_BUCKETS:
            return

        if MAX_ROWS_PER_BUCKET is not None:
            if self.counts.get(bucket, 0) >= MAX_ROWS_PER_BUCKET:
                return

        path = self._path_for_bucket(bucket)

        df = pd.DataFrame([row])
        if bucket not in self.header_written:
            df.to_csv(path, index=False, mode="w", encoding=CSV_ENCODING)
            self.header_written.add(bucket)
        else:
            df.to_csv(path, index=False, mode="a", header=False, encoding=CSV_ENCODING)

        self.counts[bucket] = self.counts.get(bucket, 0) + 1

# Find verified 5/5 strict + bucketed failures
def find_verified_5of5_streaming(dec_index, merge_neighbors, out_dir: Path,
                                 pivot_lang=PIVOT_LANG,
                                 cap_per_lang=CAP_PER_LANG,
                                 max_checks_per_pivot=MAX_CHECKS_PER_PIVOT):
    verified_rows = []
    writers = BucketWriters(out_dir)

    pivots = [n for n in merge_neighbors.keys() if n.startswith(pivot_lang + ":")]

    attempt_id = 0
    for pivot in pivots:
        needed = [l for l in LANGS if l != pivot_lang]
        cand = {l: list(merge_neighbors.get(pivot, {}).get(l, [])) for l in needed}

        if any(len(cand[l]) == 0 for l in needed):
            continue

        for l in needed:
            if len(cand[l]) > cap_per_lang:
                cand[l] = cand[l][:cap_per_lang]

        # Build product order depending on pivot lang
        # Build nodes dict with keys ar,bg,el,fr,it
        order_langs = [l for l in LANGS if l != pivot_lang]

        checked = 0
        for combo in product(*[cand[l] for l in order_langs]):
            checked += 1
            if checked > max_checks_per_pivot:
                break

            attempt_id += 1
            nodes = {pivot_lang: pivot}
            for l, n in zip(order_langs, combo):
                nodes[l] = n

            # ensure all keys exist
            # (nodes now has all 5 langs)
            ok, missing, nonmerge = verify_5tuple(dec_index, nodes)
            pairs_ok, bucket = pairs_ok_bucket(missing, nonmerge)

            if ok:
                verified_rows.append({
                    "attempt_id": attempt_id,
                    "ar_row_id": safe_row_id(nodes["ar"]),
                    "bg_row_id": safe_row_id(nodes["bg"]),
                    "el_row_id": safe_row_id(nodes["el"]),
                    "fr_row_id": safe_row_id(nodes["fr"]),
                    "it_row_id": safe_row_id(nodes["it"]),
                    "pairs_ok": 10,
                    "bucket": "10/10"
                })
            else:
                # write failure row to bucket file (streaming)
                writers.write_row(bucket, {
                    "attempt_id": attempt_id,
                    "ar_row_id": safe_row_id(nodes["ar"]),
                    "bg_row_id": safe_row_id(nodes["bg"]),
                    "el_row_id": safe_row_id(nodes["el"]),
                    "fr_row_id": safe_row_id(nodes["fr"]),
                    "it_row_id": safe_row_id(nodes["it"]),
                    "pairs_ok": pairs_ok,
                    "bucket": bucket,
                    "missing_pairs": "|".join(missing) if missing else "",
                    "nonmerge_pairs": "|".join([x["pair"] for x in nonmerge]) if nonmerge else "",
                    "nonmerge_details": " || ".join([
                        f'{x["pair"]}:{x["decision"]}:{x["confidence"]}:{x["alignment_id"]}:{(x["rationale_en"] or "")}'
                        for x in nonmerge
                    ]) if nonmerge else ""
                })

    df_ok = pd.DataFrame(verified_rows).drop_duplicates()
    return df_ok, writers.counts

# MAIN
def run(jsonl_dir: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading JSONL decisions + building indexes...")
    dec_index, merge_neighbors = build_indexes(jsonl_dir)
    print("Decision index size:", len(dec_index))
    print("Merge-neighbor nodes:", len(merge_neighbors))

    print(f"Finding VERIFIED 5/5 (10/10) with pivot={PIVOT_LANG} ...")
    df_ok, bucket_counts = find_verified_5of5_streaming(
        dec_index, merge_neighbors, out,
        pivot_lang=PIVOT_LANG,
        cap_per_lang=CAP_PER_LANG,
        max_checks_per_pivot=MAX_CHECKS_PER_PIVOT
    )

    df_ok.to_csv(out / "verified_global_5of5.csv", index=False, encoding=CSV_ENCODING)
    print("Verified 5/5:", len(df_ok))

    if WRITE_AUDIT:
        print("Audit written per bucket (rows per bucket):")
        for b in sorted(bucket_counts.keys(), key=lambda x: int(x.split("/")[0]), reverse=True):
            print(f"  {b}: {bucket_counts[b]}")
        print("Audit files are in:", str(out))

    print("DONE.")

if __name__ == "__main__":
    run(
        jsonl_dir=r"PATH\Analysis_Resutls\data_files\jsonl_files",
        out_dir=r"PATH\Alignment_Statistics_Results\csv_files\strict_union_with_pivot_it"
    )