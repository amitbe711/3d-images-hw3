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

### Hugging Face model id (`SD_MODEL_ID`)

**Leave `SD_MODEL_ID` unset on Colab** unless you need to override. `guidance/sd.py` defaults to **`sd2-community/stable-diffusion-2-1-base`**. If you still see **`stabilityai/stable-diffusion-2-1-base`** in errors, your copy is stale: run **`git pull`** in the clone (or re-clone), then **Runtime → Restart session** and rerun from the clone step.

**Bootstrap without `git pull`:** current `main.py` sets a safe default Hub id, adds **`--sd_model_id`**, and may patch stale `guidance/sd.py`. **`guidance/__init__.py`** does the same patch **before** `sd.py` is imported, so it still runs if your Colab notebook never picked up the newer `main.py` (as long as you **`git pull`** once to get `__init__.py` on disk).

**Do not** set `SD_MODEL_ID` to `stabilityai/...` in a notebook cell when that repo returns **404** for you — that forces the failing id. (If a cell already set it, `os.environ.pop("SD_MODEL_ID", None)` and restart the runtime.)

If your **grader or machine** can reach the official repo and you want that id explicitly:

```python
import os
os.environ["SD_MODEL_ID"] = "stabilityai/stable-diffusion-2-1-base"
```

Set this **before** any `import guidance.sd` or `python main.py` in that process.

---

### Private GitHub clone (optional)

`REPO_URL = f"https://YOUR_TOKEN@github.com/{GITHUB_USER}/{REPO_NAME}.git"`

---

## Step 3 — Install and fix image paths

Colab already includes **PyTorch**. Do **not** use the course **`requirements.txt`** there ( **`triton` / `xformer`** and old pins break Colab).

**`requirements-colab.txt`** uses **`diffusers>=0.31`**, **`huggingface-hub>=0.33`**, and **`accelerate>=0.32`** so imports match Colab’s **gradio / peft / datasets** stack. (`peft` imports **`clear_device_cache`** from **`accelerate.utils.memory`**, which only exists from **accelerate 0.32** upward; an older **accelerate** triggers the traceback you saw.) Older **`diffusers==0.19.3` + `hub<0.20`** avoids `cached_download` removal but **downgrades hub** and triggers resolver **conflicts** with those Colab packages.

```python
import json, pathlib

root = pathlib.Path("/content/hw3-course/HW_3")
p = root / "data" / "prompt_img_pairs.json"
data = json.loads(p.read_text())
for v in data.values():
    v["img_path"] = v["img_path"].replace("$HOME", str(root))
p.write_text(json.dumps(data, indent=4))
print("Patched:", p)

%cd /content/hw3-course/HW_3

# Colab torch 2.11 warns if setuptools is too new; keep below 82.
!pip install -q -U pip wheel
!pip install -q "setuptools>=70,<82"

# Optional: silences "ipython requires jedi" (harmless if skipped).
# !pip install -q jedi

!pip install -q -r requirements-colab.txt
# Ensure accelerate is new enough for Colab's preinstalled peft (clear_device_cache).
!pip install -q "accelerate>=0.32,<1"

# OpenAI CLIP — only for eval.py on Colab. Skip if you run eval on your laptop.
# !pip install -q git+https://github.com/openai/CLIP.git
```

**About the long `ERROR: pip's dependency resolver...` block:** if the cell **finishes** and `pip` **exit code is 0**, those lines are often **warnings** about packages pip did not reconcile globally. After **`requirements-colab.txt`** matches Colab’s hub/transformers range, most **gradio / peft / datasets** conflicts should **disappear**. If anything still breaks at **import** or **runtime**, use **Runtime → Restart session** and run **only** this notebook’s cells (or use a **fresh** Colab notebook without extra `pip install` stacks).

If **CLIP** fails, scroll up for the first `error:` line. You can skip CLIP on Colab and run **`eval.py`** locally.

If you only need **SDS/PDS**, skip the CLIP line entirely.

---

## Hugging Face: `401` / `Repository Not Found` / **404 on the model page**

The weights are downloaded from the **Hugging Face Hub** on first run.

### `401` / *Invalid username or password*

1. **A bad token is in the environment** (Colab **Secrets** → `HF_TOKEN` typo, expired token, or **Notebook access** turned off). The Hub may respond with **Repository Not Found** even when the repo exists.

2. **Gated model:** log in and accept the license on the model page (if the repo is visible to you).

**Fix for `401`:** use a valid **Read** token, secret name exactly **`HF_TOKEN`**, **Notebook access** ON for this notebook, and accept the license if the model page loads. Optional one-shot login:

```python
from huggingface_hub import login
from getpass import getpass
login(token=getpass("HF read token: "))
```

If a **broken** token is set, clear it and restart the runtime:

```python
import os
for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(k, None)
```

### **404** on `https://huggingface.co/stabilityai/stable-diffusion-2-1-base`

The course **README** may still name **`stabilityai/stable-diffusion-2-1-base`**. This repo’s **`guidance/sd.py`** defaults to **`sd2-community/stable-diffusion-2-1-base`** so Colab can download without that org page. To match the original Hub id when it works for you, set **`SD_MODEL_ID`** (see the cell above Step 3).

If the **stabilityai** page shows **404**, try logging in, another network, or ask **course staff** about access. You can also set **`SD_MODEL_ID`** to another public mirror with the same diffusers layout (e.g. **`Manojb/stable-diffusion-2-1-base`**). Confirm with your **TA** if they require the exact **`stabilityai/...`** id for grading.

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

---

## Local smoke test (optional, before Colab)

From `HW_3/` with Python 3.10+ (Colab uses 3.12; both work with `requirements-colab.txt`):

```bash
cd HW_3
python3 -m venv .venv_hw3_test
source .venv_hw3_test/bin/activate   # Windows: .venv_hw3_test\Scripts\activate
pip install -U pip setuptools wheel
pip install torch torchvision
pip install -r requirements-colab.txt
python -m py_compile guidance/sd.py main.py eval.py utils.py
python -c "import diffusers, transformers; import guidance.sd; print('OK', diffusers.__version__)"
```

This does **not** download Stable Diffusion weights (no GPU minutes). A full `python main.py ...` run pulls the configured Hub id (default **`sd2-community/stable-diffusion-2-1-base`**, or **`SD_MODEL_ID`** if set) from Hugging Face (~several GB).

**Note:** `requirements-colab.txt` uses **newer diffusers** than the course `requirements.txt` so it fits Colab’s preinstalled stack. If the grader uses a **strict** pinned conda env, confirm with staff; behavior of SDS/PDS on SD2.1 should match.
