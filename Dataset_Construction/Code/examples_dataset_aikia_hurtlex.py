import os
import re
import io, builtins, yaml, unicodedata
import snowballstemmer
import pandas as pd
from pathlib import Path

# print("Working dir:", os.getcwd())

# Load Datasets
aikia_file_path  = Path(r"C:\Users\changeme\Documents\AssignmentsForMaster\Thesis\updated_corpus_gold_corrected.tsv")
hurtlex_file_path = Path(r"C:\Users\changeme\Documents\AssignmentsForMaster\Thesis\lexicon.tsv")
hurtlex_corrected_el_path = Path(r"C:\Users\changeme\Documents\AssignmentsForMaster\Thesis\corrected_hurtlex_el.txt")

# Load HurtLex
hurtlex_df = pd.read_csv(hurtlex_file_path, sep="\t", encoding="utf-8")
print("=== HurtLex ===")
print("Path:", hurtlex_file_path)
print("Columns:", hurtlex_df.columns.tolist())
print("Shape: rows =", hurtlex_df.shape[0], ", cols =", hurtlex_df.shape[1])
print("First 5 rows:")
print(hurtlex_df.head(5).to_string(index=False))
print()

# Load AIKIA
aikia_df = pd.read_csv(aikia_file_path, sep="\t", encoding="utf-8")
print("=== AIKIA ===")
print("Path:", aikia_file_path)
print("Columns:", aikia_df.columns.to_list())
print("Shape rows=", hurtlex_df.shape[0], ", cols =", hurtlex_df.shape[1])
print("First 5 rows:")
print(aikia_df.head(5).to_string(index=False))
print()

# Load HurtLex corrected el
hurtlex_corrected_el_df = pd.read_csv(hurtlex_corrected_el_path, sep="\t", encoding="utf-8")
print("=== Hurtlex corrected el ===")
print("Path:", hurtlex_corrected_el_path)
print("Columns:", hurtlex_corrected_el_df.columns.tolist())
print("Shape: rows =", hurtlex_corrected_el_df.shape[0], ", cols =", hurtlex_corrected_el_df.shape[1])
print("First 5 rows:")
print(hurtlex_corrected_el_df.head(25).to_string(index=False))
print()

# Check which words are in actual HurtLex Repository
# Preprocess dataset
# Reorder corrected DataFrame
hurtlex_corrected_el_df.rename(columns={"lemma corrected hurtlex": "lemma original hurtlex", "lemma original hurtlex":  "lemma corrected hurtlex"}, inplace=True)
print("Columns:", hurtlex_corrected_el_df.columns.tolist())

#out_path = Path("hurtlex_corrected_file_el.txt")
#hurtlex_corrected_el_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")

corr_words = hurtlex_corrected_el_df["lemma corrected hurtlex"].tolist()
plain_words = hurtlex_df["Lexicon"].tolist()
max_len = max(len(corr_words), len(plain_words))
corr_pad  = corr_words  + [""] * (max_len - len(corr_words))
plain_pad = plain_words + [""] * (max_len - len(plain_words))
compare_df = pd.DataFrame({"corrected_hurtlex_el": corr_pad, "plain_hurtlex": plain_pad})

#out_path = Path("compare_corrected_vs_plain.txt")
#compare_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
#print(f"Written comparison file → {out_path}")

# Remove accents from a string and lowercase
def remove_accents(text: str) -> str:
    nfkd = unicodedata.normalize('NFD', text)
    without_marks = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return unicodedata.normalize('NFC', without_marks)

#hurtlex_corrected_el_df["lemma corrected hurtlex"] = hurtlex_corrected_el_df["lemma corrected hurtlex"].apply(remove_accents).str.lower()
#print(f"Loaded lexicon: {hurtlex_corrected_el_df.shape[0]} rows → first 5 clean entries:\n", hurtlex_corrected_el_df[["lemma corrected hurtlex", "lemma original hurtlex"]].head(), "\n")

aikia_df.columns = ["id", "sentence", "score"]
aikia_df["sentence_clean"] = aikia_df["sentence"].apply(remove_accents).str.lower()
print(f"Loaded AIKIA shape: {aikia_df.shape[0]} rows → first 5 clean entries:\n", aikia_df[["sentence", "sentence_clean"]].head(), "\n")


# ------------------------------------------------------------------
# UTF-8 (Windows console = cp1253 by default)
# ------------------------------------------------------------------
_orig_open = builtins.open
def _utf8_open(file, *a, **kw):
    kw.setdefault("encoding", "utf-8")     
    return _orig_open(file, *a, **kw)

builtins.open = _utf8_open
io.open       = _utf8_open 

# Patched yaml.load
_orig_yaml_load = yaml.load
def _patched_yaml_load(stream, Loader=None, *a, **kw):
    if Loader is None:                      
        return yaml.safe_load(stream)
    return _orig_yaml_load(stream, Loader, *a, **kw)
yaml.load = _patched_yaml_load

from greek_stemmer import GreekStemmer  

# Initialize stemmers
sb = snowballstemmer.stemmer("greek")            
gs = GreekStemmer()

# Apply stemmers
#hurtlex_corrected_el_df["snowball_stem"] = hurtlex_corrected_el_df["lemma corrected hurtlex"].apply(lambda w: sb.stemWords([w])[0])
#hurtlex_corrected_el_df["greek_stemmer_stem"] = hurtlex_corrected_el_df["lemma corrected hurtlex"].apply(gs.stem)

# Save results for comparison
#out_path = Path("hurtlex_stems.tsv")
#hurtlex_corrected_el_df[["lemma corrected hurtlex", "snowball_stem", "greek_stemmer_stem"]].to_csv(out_path, sep="\t", index=False, encoding="utf-8")
#print(f"✓ Stems written to {out_path}")

#  Find the Sentences

# Load stems
path = r"C:\Users\changeme\Documents\AssignmentsForMaster\Thesis\hurtlex_stems_corrected_book1.txt"
df = pd.read_csv(path, sep="\t", skiprows=1)

print(df.head())
print(df.columns)

# Find the matches sentences
results = []

for _, row in df.iterrows():
    lemma = row['lemma corrected hurtlex']
    stem = row['corrected_snowball_stem']
    pattern = r'\b' + re.escape(stem)  
    matches = aikia_df[aikia_df['sentence_clean'].str.contains(pattern, na=False, regex=True)]
    for _, match_row in matches.iterrows():
        results.append({
            'lemma': lemma,
            'stem': stem,
            'id': match_row['id'],
            'sentence': match_row['sentence'],
            'score': match_row['score'],
        })

results_df = pd.DataFrame(results)
results_df.to_csv('matched_sentences_v1.csv', index=False, encoding='utf-8')