# HW3 on Google Colab (git workflow)

Use a **GPU runtime** (Runtime → Change runtime type → T4 GPU).

## Why your Colab cell failed

In bash, **`<` and `>` are redirection**, not placeholders.

So a line like:

`!git clone https://github.com/<you>/<repo>.git hw3-course`

makes bash try to read from a file named `you` → **`you: No such file or directory`**.

The clone never ran, so **`/content/hw3-course/HW_3` does not exist**.

**Fix:** replace `YOUR_GITHUB_USER` and `YOUR_REPO_NAME` below with your real values **without** any `<` `>` characters.

---

## Step 1 — Push this repo from your laptop

From the course root (`3D Images`), after commits are on `main`:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USER/YOUR_REPO_NAME.git
git push -u origin main
```

Use HTTPS + GitHub login, or an SSH remote. Do not commit secrets (`.env`, tokens).

---

## Step 2 — Clone in Colab (copy-paste)

**Edit the two variables** in the first cell, then run all cells in order.

```python
# --- EDIT THESE (no angle brackets!) ---
GITHUB_USER = "YOUR_GITHUB_USER"   # e.g. "amitbenbenishti"
REPO_NAME = "YOUR_REPO_NAME"       # e.g. "3d-images-course"
# --------------------------------------

import os
CLONE_DIR = "/content/hw3-course"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git"

%cd /content
!rm -rf {CLONE_DIR}   # optional: clean re-clone
!git clone {REPO_URL} {CLONE_DIR}

HW3 = f"{CLONE_DIR}/HW_3"
assert os.path.isdir(HW3), f"Clone OK but HW_3 not found at {HW3}. Is HW_3 at repo root?"
%cd {HW3}
!pwd
!ls -la
```

Private repo: use a [fine-grained PAT](https://github.com/settings/tokens) as the password when `git` prompts, or embed **only for the session** (avoid saving in the notebook):

`REPO_URL = f"https://YOUR_TOKEN@github.com/{GITHUB_USER}/{REPO_NAME}.git"`

---

## Step 3 — Install and fix image paths

Colab already includes **PyTorch**. The course `requirements.txt` pins **`triton`** and **`xformer`**, which often break pip with **Getting requirements to build wheel**. Use the Colab file instead, then install **CLIP** in a second step (for `eval.py` only).

```python
import json, pathlib

root = pathlib.Path("/content/hw3-course/HW_3")
p = root / "data" / "prompt_img_pairs.json"
data = json.loads(p.read_text())
for v in data.values():
    v["img_path"] = v["img_path"].replace("$HOME", str(root))
p.write_text(json.dumps(data, indent=4))
print("Patched:", p)

# HW3 main.py / SDS / PDS (no triton / xformer)
!pip install -q -U pip setuptools wheel
!pip install -q -r requirements-colab.txt

# OpenAI CLIP — only needed for eval.py; second line avoids dragging triton from full requirements
!pip install -q git+https://github.com/openai/CLIP.git
```

If pip fails with **Getting requirements to build wheel**, scroll **up** in the cell output: the failing package is usually **`tokenizers`** when Colab is on **Python 3.12** and an old **`transformers` / `tokenizers`** pin forces a Rust build. The repo’s **`requirements-colab.txt`** avoids that by using **`transformers>=4.36`** (binary wheels). Do **not** run the course **`requirements.txt`** on Colab unless you use a Python 3.10/3.11 environment.

If **CLIP** still fails on a very new Colab Python, scroll up in the log for the **first** `error:` block (often `sentencepiece` or `clip` `setup.py`). Try:

```text
!pip install -q "setuptools>=64" "packaging>=23"
!pip install -q git+https://github.com/openai/CLIP.git
```

If you only need to **run SDS/PDS** and will run **`eval.py` on your laptop**, you can skip the CLIP line entirely.

---

## Step 4 — Smoke test (SDS)

```python
!python main.py --prompt "A dog sitting on grass" --loss_type sds --guidance_scale 25
```

---

## Step 5 — After Colab: keep outputs

Colab disks are ephemeral. Either:

- Download `outputs/` from the Colab file browser, or  
- Zip to Drive:

```python
!cd /content/hw3-course/HW_3 && zip -r /content/hw3_outputs.zip outputs
from google.colab import files
files.download("/content/hw3_outputs.zip")
```

Put extracted outputs on your laptop under `HW_3/outputs/` for the course zip. That folder is **gitignored** on purpose.

---

## Step 6 — Laptop: commit code changes only

```bash
git add HW_3/guidance/sd.py
git commit -m "Describe change"
git push
```

Do not commit large generated PNG batches unless you really want them in Git history.
