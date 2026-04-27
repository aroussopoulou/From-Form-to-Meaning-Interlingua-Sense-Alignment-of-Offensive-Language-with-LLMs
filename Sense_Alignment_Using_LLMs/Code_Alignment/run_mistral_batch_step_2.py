import os, time, json, pathlib, tqdm 
from mistralai import Mistral

"""
Uploads every batch_x.jsonl, launches one job per file, waits for completion, downloads outputs, and merges them.
"""
# Config
API_KEY    = "API_Key" 
MODEL      = "mistral-small-latest"       
BATCH_DIR  = pathlib.Path(r"PATH\alignments_files\files_jsonl\First_Batches\batches_el_fr")
MERGE_OUT  = pathlib.Path(r"PATH\alignments_files\files_jsonl\Total\alignment_elfr.jsonl")
POLL_SECS  = 30

client = Mistral(api_key=API_KEY)

jobs = []                                           

for jf in tqdm.tqdm(sorted(BATCH_DIR.glob("batch_*.jsonl")), desc="upload/start"):
    with jf.open("rb") as f:
        fid = client.files.upload(
            file={"file_name": jf.name, "content": f},
            purpose="batch"
        ).id
    jid = client.batch.jobs.create(
            endpoint="/v1/chat/completions",
            input_files=[fid],
            model=MODEL
         ).id
    jobs.append((jid, jf.with_suffix(".out.jsonl")))

for jid, outp in tqdm.tqdm(jobs, desc="waiting/downloading"):
    while True:
        job = client.batch.jobs.get(job_id=jid)          
        if job.status == "succeeded":
            break
        if job.status == "failed":
            raise RuntimeError(f"Job {jid} failed: {job.error}")
        time.sleep(POLL_SECS)

    with outp.open("wb") as fout:                       
        fout.write(client.files.download(file_id=job.output_file).read())


with MERGE_OUT.open("w", encoding="utf-8") as fout:
    for _, outp in jobs:
        with outp.open(encoding="utf-8") as fin:
            for ln in fin:
                rec  = json.loads(ln)
                cid  = rec["custom_id"]
                data = json.loads(rec["response"]["choices"][0]["message"]["content"])
                fout.write(json.dumps({
                    "row_id":          cid,
                    "decision":        data["decision"],
                    "confidence":      data["confidence"],
                    "sense_id":        data["globalsense_id"],
                    "sense_label_en":  data["globalsense_label_en"],
                    "rationale_en":    data["rationale_en"],
                }, ensure_ascii=False) + "\n")

print("✓ Τελείωσε – συγχωνευμένο αρχείο:", MERGE_OUT)
