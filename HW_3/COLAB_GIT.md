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

Some accounts or regions report that the **Stability AI** repo page returns **404** (or the org made access stricter). Your code no longer depends only on that id: by default it loads a **public community mirror** with the same **SD 2.1 base** diffusers layout:

- Default: **`sd2-community/stable-diffusion-2-1-base`**

To force the original id (if it works for you):

```python
import os
os.environ["SD_MODEL_ID"] = "stabilityai/stable-diffusion-2-1-base"
```

Or in Colab before `main.py`:

```text
!SD_MODEL_ID=stabilityai/stable-diffusion-2-1-base python main.py ...
```

Other mirrors people use: **`Manojb/stable-diffusion-2-1-base`** (set `SD_MODEL_ID` the same way). Prefer mirrors that expose the full **diffusers** file tree (`model_index.json`, `scheduler/`, etc.).

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

This does **not** download Stable Diffusion weights (no GPU minutes). A full `python main.py ...` run downloads the **default Hub repo** (see `SD_MODEL_ID` / `sd.py`; often **several GB**).

**Note:** `requirements-colab.txt` uses **newer diffusers** than the course `requirements.txt` so it fits Colab’s preinstalled stack. If the grader uses a **strict** pinned conda env, confirm with staff; behavior of SDS/PDS on SD2.1 should match.
