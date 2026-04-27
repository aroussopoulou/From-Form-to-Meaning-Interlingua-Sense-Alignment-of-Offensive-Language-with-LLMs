from __future__ import annotations
import json, re
from itertools import combinations, product
from pathlib import Path
import pandas as pd


# ───── PATHS ─────
HURTNET_CSV  = Path(r"PATH\Alignment_File\hurtnet_total_dataset.csv")
TSV_ALL5_CSV = Path(r"PATH\Alignment_Statistics_Results\csv_files\Comparison_per_HurtLex_and_HurtNet\tsv_global_merges_all5_in_hurtnet.csv")
ALIGN_DIR    = Path(r"PATH\Alignment_Statistics_Results\jsonl_files")

OUT_DIR = Path(r"PATH\Alignment_Statistics_Results\csv_files\Comparison_per_HurtLex_and_HurtNet")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CLUSTER = OUT_DIR / "cluster_level_alignment_polysemy_decisions_rationales_version_5.csv"
OUT_DEBUG   = OUT_DIR / "debug_implied_pairs_with_decisions_version_5.csv"

LANGS = ["ar","bg","el","fr","it"]
LANGNAME2CODE = {"Arabic":"ar","Bulgarian":"bg","Greek":"el","French":"fr","Italian":"it"}
TSV_COLS = {
    "ar":["AR_ids_set","AR_ids"], "bg":["BG_ids_set","BG_ids"],
    "el":["EL_ids_set","EL_ids"], "fr":["FR_ids_set","FR_ids"],
    "it":["IT_ids_set","IT_ids"],
}

# ───── helpers ─────
def pick_col(df,cands):
    for c in cands:
        if c in df.columns: return c
    return None
def split_ids(cell):
    if cell is None or (isinstance(cell,float) and pd.isna(cell)): return []
    s=str(cell).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try: return sorted(set(str(x).strip() for x in json.loads(s)))
        except Exception: pass
    return sorted(set(v.strip() for v in re.split(r"[|,]\s*",s) if v.strip()))
def pkey(l1,r1,l2,r2):
    a,b=(l1,r1),(l2,r2)
    return (a,b) if a<=b else (b,a)

# 1. Hurtnet
hur=pd.read_csv(HURTNET_CSV)
hur=hur[hur["Language"].isin(LANGNAME2CODE)]
hur["lang"]=hur["Language"].map(LANGNAME2CODE)
hur["Row_id"]=pd.to_numeric(hur["Row_id"],errors="coerce").astype("Int64")
hur=hur.dropna(subset=["ID","Lemma","Row_id","lang"])
id_info=hur[["ID","lang","Row_id","Lemma"]].drop_duplicates("ID").set_index("ID")
poly={(r.lang,r.Lemma):int(r.n_senses)
      for r in hur.groupby(["lang","Lemma"])["Row_id"].nunique()
               .reset_index(name="n_senses").itertuples(False)}

# 2. TSV clusters
tsv=pd.read_csv(TSV_ALL5_CSV)
if "alignment id" in tsv.columns and "alignment_id" not in tsv.columns:
    tsv=tsv.rename(columns={"alignment id":"alignment_id"})
col_lang={L:pick_col(tsv,TSV_COLS[L]) for L in LANGS}

clusters=[]
for _,r in tsv.iterrows():
    ids_by={}
    for L in LANGS:
        ids=[x for x in split_ids(r[col_lang[L]])
                if x in id_info.index and id_info.loc[x,"lang"]==L]
        if not ids: break
        ids_by[L]=sorted(set(ids))
    else:
        clusters.append({"alignment_id":r["alignment_id"],"ids_by":ids_by})
cl_df=pd.DataFrame(clusters)
if cl_df.empty: raise ValueError("No valid clusters.")

# 3. JSONL lookup
lookup={}
for jf in ALIGN_DIR.glob("alignment_*.jsonl"):
    with jf.open(encoding="utf-8") as f:
        for ln in f:
            try: obj=json.loads(ln.strip())
            except Exception: continue
            rows=[(k[:-7],obj[k]) for k in obj if k.endswith("_row_id")]   # ← FIX εδώ
            if len(rows)!=2: continue
            (l1,r1),(l2,r2)=rows
            if l1 not in LANGS or l2 not in LANGS: continue
            try: r1=int(r1); r2=int(r2)
            except Exception: continue
            pay=obj.get("response") or obj
            dec = pay.get("decision")
            conf= pay.get("confidence")
            rat = pay.get("rationale_en") or pay.get("rationale") or pay.get("reason")
            cv=float(conf) if conf not in (None,"") else -1.0
            k=pkey(l1,r1,l2,r2)
            if k not in lookup or cv>lookup[k]["cv"]:
                lookup[k]={"decision":dec,"confidence":conf,"cv":cv,"rationale":rat}

# 4. build outputs
cl_out=[]; dbg=[]
for _,cl in cl_df.iterrows():
    aid=cl["alignment_id"]; ids_by=cl["ids_by"]
    row={"alignment_id":aid}
    for L in LANGS:
        ids=ids_by[L]
        lemmas=sorted({id_info.loc[i,"Lemma"] for i in ids})
        row[f"{L}_ids_set"]=json.dumps(ids,ensure_ascii=False)
        row[f"{L}_lemmas_set"]=json.dumps(lemmas,ensure_ascii=False)
        row[f"{L}_poly"]=json.dumps({lm:poly.get((L,lm),1) for lm in lemmas},ensure_ascii=False)

    dcount={}; ratlist=[]; tot=found=0
    for L1,L2 in combinations(LANGS,2):
        for id1,id2 in product(ids_by[L1],ids_by[L2]):
            r1=int(id_info.loc[id1,"Row_id"]); r2=int(id_info.loc[id2,"Row_id"])
            rec=lookup.get(pkey(L1,r1,L2,r2))
            dec = rec["decision"] if rec else "not_found"
            rat = rec["rationale"] if rec else None
            dcount[dec]=dcount.get(dec,0)+1
            ratlist.append({"langpair":f"{L1}-{L2}","id1":id1,"id2":id2,
                            "decision":dec,"rationale_en":rat})
            dbg.append({"alignment_id":aid,"langpair":f"{L1}-{L2}",
                        "id1":id1,"id2":id2,"row_id1":r1,"row_id2":r2,
                        "decision":dec,"rationale_en":rat})
            tot+=1; found+=bool(rec)

    row["n_implied_pairs_total"]=tot
    row["n_pairs_found"]=found
    row["decision_counts"]=json.dumps(dcount,ensure_ascii=False)
    row["rationale_en_list"]=json.dumps(ratlist,ensure_ascii=False)
    cl_out.append(row)

df=pd.DataFrame(cl_out)
df.to_csv(OUT_CLUSTER,index=False)
pd.DataFrame(dbg).to_csv(OUT_DEBUG,index=False)
print("✅ Τέλος – έτοιμα CSV:\n",OUT_CLUSTER,"\n",OUT_DEBUG)