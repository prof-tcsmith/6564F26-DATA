# Getting started on your own machine

*ISM 6564 · Text Analytics · running the course notebooks locally with Miniconda, VS Code, and your GPU*

Everything in this course also runs in Google Colab, and the Week 5 lab opens there
with one click. This guide is for running on your own laptop instead: it is faster to
iterate, it works offline once the models are downloaded, your data never leaves your
machine, and Assignment 5 asks you to measure *your* hardware. Budget 30 to 45 minutes
the first time, most of it downloads.

> **Already ran `./setup.sh` from the course data repository in Week 1?** That built the
> same environment with `uv`. Skip to **Step 6** to check your GPU and open the lab.
> On **Windows with an NVIDIA GPU**, `setup.sh` installed the CPU-only PyTorch: swap in
> the NVIDIA build with the uv line in Step 3 first.
> This guide is the step-by-step alternative for anyone who prefers conda or wants to
> see each piece.

> **Test your install with the setup notebook.** `ism6564-local-llm-setup.ipynb` (on
> Canvas) checks your Python, PyTorch and GPU, has a small model write and train, and ends
> with a PASS / WARN / FAIL summary. It also walks through the Windows (NVIDIA) and Mac
> setups in more detail, with links to the official guides. Getting your own machine
> working is your responsibility; the instructor does not provide tech support for personal
> installs, so start early, and use Colab while you fix it.

## What you are installing, and why

| Piece | What it does |
| --- | --- |
| **Miniconda** | Installs Python and keeps this course's packages in their own environment, so nothing here collides with anything else on your machine. |
| **PyTorch** | The numerical engine every model runs on. It comes in three builds: one that uses an NVIDIA GPU (CUDA), one that uses the GPU inside Apple Silicon Macs (MPS), and a CPU-only build. |
| **transformers, sentence-transformers** | Hugging Face's libraries: download a model by name, run it. |
| **VS Code + the Python and Jupyter extensions** | The editor, and the notebook runner inside it. |

## Step 1 — Miniconda

1. Download the installer for your machine from
   [anaconda.com/docs/getting-started/miniconda/main](https://www.anaconda.com/docs/getting-started/miniconda/main).
   - **Mac:** pick *Apple Silicon* if your Mac has an M-series chip (Apple menu → *About This Mac* says "Chip: Apple M…"), otherwise *Intel*. Use the `.pkg` installer.
   - **Windows:** the 64-bit `.exe`. Keep the defaults. Leave *Add Miniconda3 to my PATH* **unchecked**; you will use the terminal the installer sets up.
2. Open a terminal that knows about conda:
   - **Mac:** open **Terminal**, run `conda init zsh`, then close and reopen Terminal.
   - **Windows:** open **Anaconda Prompt (miniconda3)** from the Start menu. To make conda work in PowerShell too, run `conda init powershell` once and reopen PowerShell.
3. Check:

   ```bash
   conda --version
   ```

## Step 2 — Create the course environment

```bash
conda create -n ism6564 python=3.12 -y
conda activate ism6564
```

Your prompt now starts with `(ism6564)`. Every `pip install` from here goes into this
environment only. Whenever you open a new terminal for this course, run
`conda activate ism6564` first.

## Step 3 — PyTorch, with your GPU if you have one

PyTorch is installed with pip; the PyTorch team no longer publishes conda packages.
Pick **one** of the three commands.

**Mac with Apple Silicon (M1 or later), on macOS 14 or later.** The standard build includes
MPS, Apple's GPU backend. Your Python must be an Apple-silicon (arm64) build: an Intel Python
running under Rosetta cannot see the GPU (`python -c "import platform; print(platform.machine())"`
must print `arm64`).

```bash
pip install torch torchvision torchaudio
```

**Windows with an NVIDIA GPU.** First check the driver:

```bash
nvidia-smi
```

If it prints a table, look at *CUDA Version* in the top right; you need a driver that
supports CUDA 12 or later. If the command is not found, update the driver from
nvidia.com first. Then install the CUDA build. The wheel contains its own CUDA runtime,
so you do not install the CUDA toolkit separately:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

`cu126` suits most cards, including GTX 10-series and older ones. A **GeForce RTX 50-series**
card needs a newer build: use `cu130` in place of `cu126` (it needs driver 580 or newer). If the
selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) shows a
different CUDA index for your driver, use that URL; the pattern is the same.

**Built the environment with `./setup.sh` (uv) instead?** Run this in your `6564F26-DATA`
folder to swap in the NVIDIA build. `./setup.sh`, `uv sync` and `uv run` put the CPU-only
build back, so run it again after any of them:

```bash
uv pip install --python .venv --reinstall-package torch torch --index-url https://download.pytorch.org/whl/cu126
```

**Everything else** (Windows without an NVIDIA GPU, Linux without NVIDIA). The CPU build. An
**Intel Mac** cannot run current PyTorch (Intel-Mac builds stopped at 2.2.2), so use Colab:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

CPU is fine for every notebook in this course; the half-billion-parameter models
answer at a few tokens per second. For anything heavier, use Colab's free T4.

**Verify:**

```bash
python -c "import torch; print(torch.__version__, '| cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"
```

You want `cuda True` on an NVIDIA machine, `mps True` on an Apple Silicon Mac, and
`False False` otherwise.

## Step 4 — The rest of the packages

```bash
pip install transformers sentence-transformers accelerate datasets peft sentencepiece huggingface-hub scikit-learn pandas matplotlib jupyter ipykernel ipywidgets
python -m ipykernel install --user --name ism6564 --display-name "Python (ism6564)"
```

The second line registers the environment as a notebook kernel, so VS Code and Jupyter
can find it by name.

## Step 5 — VS Code

1. Install VS Code from [code.visualstudio.com](https://code.visualstudio.com).
2. Extensions (left bar → *Extensions*): install **Python** (`ms-python.python`) and **Jupyter** (`ms-toolsai.jupyter`), both by Microsoft.
3. *File → Open Folder* and open the folder where you keep course notebooks.
4. Open a `.ipynb`. Top right, click **Select Kernel → Python Environments → Python (ism6564)**. If it is not listed: *View → Command Palette → "Python: Select Interpreter"*, pick the `ism6564` environment, then *Developer: Reload Window*.
5. Run the first cell. The kernel name shows top right; the output shows which device you are on.

## Step 6 — Confirm the GPU from a notebook

Paste into a cell and run it:

```python
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
```

Every course notebook makes this same three-way choice, so nothing else changes
between a CUDA machine, a Mac and a CPU-only laptop. For the full test, run the setup
notebook, `ism6564-local-llm-setup.ipynb` (on Canvas). For scale, at Station 3 of the
Week 5 lab (Qwen2.5-0.5B, float32, 48 tokens) the instructor's Apple Silicon Mac
produces about 50 to 70 tokens per second on MPS. A recent NVIDIA GPU is faster. A
laptop CPU manages a few tokens per second, which is enough for the lab.

## Step 7 — Where the models live

Hugging Face downloads a model once and reuses it:

| | |
| --- | --- |
| Mac, Linux | `~/.cache/huggingface/hub` |
| Windows | `C:\Users\<you>\.cache\huggingface\hub` |

Each model is a folder named `models--<org>--<name>`. Delete a folder to reclaim its
space; it downloads again on next use. To keep the cache on another drive, set the
environment variable `HF_HOME` to a folder there before starting VS Code. No Hugging
Face account or token is needed for any model this course uses.

**Windows note.** Hugging Face prints a warning about symlinks the first time. It is
harmless. To silence it, set the environment variable
`HF_HUB_DISABLE_SYMLINKS_WARNING=1`, or turn on *Developer Mode* in Windows Settings.

## Step 8 — Run the Week 5 lab

Download `lab-slm-by-shape.ipynb` from `week-05-transformers-llms/lab/` (*Raw → Save*)
into your course folder, open it in VS Code, select the `Python (ism6564)` kernel, and
run the first two cells **at home**: they fetch about 2.7 GB of models. Then work the
stations. The notebook itself has the results sheet; the timing plan is on Canvas.

## When something goes wrong

| Symptom | Fix |
| --- | --- |
| `conda: command not found` | The terminal was opened before `conda init` ran. Close it and open a new one. On Windows use *Anaconda Prompt*. |
| `ModuleNotFoundError: No module named 'torch'` inside the notebook | Wrong kernel. Select `Python (ism6564)` top right. |
| `torch.cuda.is_available()` is `False` on an NVIDIA machine | You have the CPU build. `pip uninstall torch torchvision torchaudio`, then reinstall with the `cu126` index URL (`cu130` for an RTX 50-series card). Check that `nvidia-smi` works first. With uv, use the uv line in Step 3. |
| `mps` is `False` on a Mac | An Intel Mac (use Colab); an Intel Python running under Rosetta on an M-series Mac (`platform.machine()` prints `x86_64`: install an Apple-silicon Python and rebuild the environment); or macOS older than 14 (update macOS). |
| `NotImplementedError: … not currently implemented for the MPS device` | A rare operation is missing on MPS. Set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` and restart the kernel. |
| The kernel dies at Station 3, or "out of memory" | Close other applications; restart the kernel and run Setup, Fetch and Station 3 only. On an NVIDIA GPU you can also load the model in `torch.float16`; on a Mac, do not: half precision garbles the small Qwen models there. |
| Downloads fail or stall | Run the cell again; downloads resume. Some campus and corporate networks block `huggingface.co`: use a phone hotspot, or Colab. |
| VS Code cannot see the environment | Command Palette → *Python: Select Interpreter → Enter interpreter path* → the `python` inside `…/miniconda3/envs/ism6564/` (`Scripts\python.exe` on Windows, `bin/python` on Mac). |

## Colab instead

Open the notebook from GitHub in Colab, choose a T4 runtime, run. The only
differences: models download again in each session, sessions disconnect after about
90 idle minutes, and the machine you measure in Assignment 5 is Google's rather than
yours. Say so in the write-up.
