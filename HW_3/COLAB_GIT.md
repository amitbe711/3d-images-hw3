# HW3 on Google Colab (git workflow)

Use a **GPU runtime** (Runtime → Change runtime type → T4 GPU).

## Step 1 — Push this repo from your laptop

From the course root (`3D Images`), after commits are on `main` (or your branch):

```bash
# one-time: create empty repo on GitHub, then:
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Use **HTTPS + GitHub credential** or **SSH** as you prefer. Do not commit secrets (`.env`, tokens).

## Step 2 — Clone in Colab

In a Colab cell:

```python
%cd /content
!rm -rf hw3-course  # optional: clean re-clone
!git clone https://github.com/<you>/<repo>.git hw3-course
%cd /content/hw3-course/HW_3
```

If the repo is private, use a [fine-grained PAT](https://github.com/settings/tokens) (read-only) in the URL only for that session, or use SSH with a deploy key (more setup).

## Step 3 — Install and fix paths

```python
!pip install -q -r requirements.txt

import json, pathlib
root = pathlib.Path("/content/hw3-course/HW_3")
p = root / "data" / "prompt_img_pairs.json"
data = json.loads(p.read_text())
for v in data.values():
    v["img_path"] = v["img_path"].replace("$HOME", str(root))
p.write_text(json.dumps(data, indent=4))
```

## Step 4 — Run one SDS job (smoke test)

```python
!python main.py --prompt "A dog sitting on grass" --loss_type sds --guidance_scale 25
```

## Step 5 — After Colab: pull outputs or re-push

Colab is ephemeral. Either:

- **Download** `outputs/` from the Colab file browser, or  
- **Commit from Colab** (not ideal): configure `git` + token and push a branch with outputs (large; usually worse than zipping to Drive).

Recommended: zip `outputs/` and copy to Google Drive, then on your laptop place them under `HW_3/outputs/` (gitignored) for the course zip submission.

## Step 6 — Laptop: optional commit after you merge results

Only when you want history updated:

```bash
git add HW_3/guidance/sd.py   # example
git commit -m "..."
```

Do not `git add` generated PNGs if you want a small repo; submission zips stay local per course instructions.
