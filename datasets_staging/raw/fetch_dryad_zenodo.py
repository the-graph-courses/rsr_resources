#!/usr/bin/env python3
"""Fetch Dryad/Zenodo datasets and save files to raw/ with per-dataset subfolders.

Prints a small report per dataset: license, file list, notes.
"""
import json
import os
import sys
import urllib.parse
import urllib.request

RAW = os.path.dirname(os.path.abspath(__file__))  # datasets_staging/raw

def http_get(url, as_json=False, dest=None):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if dest:
        with open(dest, "wb") as f:
            f.write(data)
        return dest
    if as_json:
        return json.loads(data)
    return data

def fetch_dryad(doi, folder_name):
    """Fetch all files from a Dryad dataset by DOI."""
    doi_enc = urllib.parse.quote(f"doi:{doi}", safe="")
    meta = http_get(f"https://datadryad.org/api/v2/datasets/{doi_enc}", as_json=True)
    version_href = meta["_links"]["stash:version"]["href"]
    version_id = version_href.rsplit("/", 1)[-1]
    files = http_get(f"https://datadryad.org/api/v2/versions/{version_id}/files", as_json=True)

    outdir = os.path.join(RAW, folder_name)
    os.makedirs(outdir, exist_ok=True)

    result = {
        "doi": doi,
        "title": meta.get("title"),
        "license": meta.get("license"),
        "sharing_link": meta.get("sharingLink"),
        "files": [],
    }
    for f in files["_embedded"]["stash:files"]:
        path = f["path"].replace("/", "_")
        url = "https://datadryad.org" + f["_links"]["stash:download"]["href"]
        dest = os.path.join(outdir, path)
        try:
            http_get(url, dest=dest)
            result["files"].append({"name": path, "size": f.get("size"), "mime": f.get("mimeType"), "url": url})
        except Exception as e:
            result["files"].append({"name": path, "error": str(e), "url": url})
    return result

def fetch_zenodo(record_id, folder_name):
    meta = http_get(f"https://zenodo.org/api/records/{record_id}", as_json=True)
    outdir = os.path.join(RAW, folder_name)
    os.makedirs(outdir, exist_ok=True)
    result = {
        "record": record_id,
        "title": meta.get("metadata", {}).get("title"),
        "license": (meta.get("metadata", {}).get("license") or {}).get("id"),
        "sharing_link": meta.get("links", {}).get("html"),
        "files": [],
    }
    for f in meta.get("files", []):
        name = f["key"]
        url = f["links"]["self"]
        dest = os.path.join(outdir, name)
        try:
            http_get(url, dest=dest)
            result["files"].append({"name": name, "size": f.get("size"), "mime": f.get("mimetype"), "url": url})
        except Exception as e:
            result["files"].append({"name": name, "error": str(e), "url": url})
    return result

TASKS = [
    ("dryad", "10.5061/dryad.0k6djhb14", "06_peak_power_bone"),
    ("dryad", "10.5061/dryad.nf63rb8",    "11_hiv_6mwt"),
    ("zenodo", "4946112",                 "12_ggt_atherosclerosis"),
    ("zenodo", "4961200",                 "13_cimt_ra"),
    ("dryad", "10.5061/dryad.ff6bd0pq",   "14_oc_prostate"),
    ("dryad", "10.5061/dryad.jg413",      "15_pam13"),
    ("dryad", "10.5061/dryad.63r07",      "16_med_student_qol"),
    ("dryad", "10.5061/dryad.bnzs7h4j1",  "17_depression_anxiety"),
]

if __name__ == "__main__":
    reports = []
    for kind, ident, folder in TASKS:
        print(f"\n=== {folder}  ({kind}: {ident}) ===", flush=True)
        try:
            if kind == "dryad":
                r = fetch_dryad(ident, folder)
            else:
                r = fetch_zenodo(ident, folder)
            r["kind"] = kind
            r["folder"] = folder
            print(f"  license: {r.get('license')}")
            print(f"  title: {r.get('title')}")
            for f in r["files"]:
                err = f" ERROR: {f['error']}" if "error" in f else ""
                print(f"  - {f['name']} ({f.get('size')} bytes, {f.get('mime')}){err}")
            reports.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            reports.append({"kind": kind, "folder": folder, "id": ident, "error": str(e)})
    with open(os.path.join(RAW, "_fetch_report.json"), "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nReport saved to {os.path.join(RAW, '_fetch_report.json')}")
