# Create executable

Build **dot** as a standalone app (no Python install required). The result is a **frozen** bundle (Python + dependencies + your code), not a native compiled binary.

## Options

| Approach | Output | Pros | Cons |
|----------|--------|------|------|
| **PyInstaller** (recommended) | Folder `dist/dot/` (or `dot.app` on macOS) | Documented, used for existing Windows/Mac builds; spec file in repo | Large size; antivirus may flag first run |
| **Nuitka** | Single executable possible | Can produce smaller, faster native binary | Harder with PyTorch/ONNX/CUDA; less common for this stack |
| **Docker** | Container image | Reproducible, no host Python | User needs Docker; GUI/device access more involved |

The codebase includes **PyInstaller** support and path handling for frozen runs (`dot.commons.paths`).

---

## Recommended: PyInstaller with the spec file

Use the provided **`dot.spec`** so you don’t need to paste `site-packages` paths.

### Prerequisites

- Python 3.9–3.12, with dot deps installed (conda or `uv`/pip).
- **saved_models** (optional): download checkpoints and put them in `saved_models/` at repo root so they are bundled. If the folder is missing, the spec still builds but models must be provided at runtime.
- **PyInstaller**: `pip install pyinstaller` or `pip install -e ".[dev]"` (dev extra includes pyinstaller).

### Build (all platforms)

From the **repo root**:

```bash
cd path/to/dot
# Activate your env (conda activate dot, or source .venv/bin/activate, etc.)
pip install -e .
pip install pyinstaller
pyinstaller dot.spec
```

- **Windows**: run `dist\dot\dot.exe`.
- **Linux**: run `dist/dot/dot`.
- **macOS**: output is `dist/dot/` (directory). To get a double‑clickable app you can rename/wrap it; then run:
  ```bash
  xattr -cr dist/dot.app
  ```
  If the camera fails, run the binary inside the app:  
  `dot.app/Contents/MacOS/dot`.

The app resolves **configs**, **data**, and **saved_models** relative to the bundle (see `src/dot/commons/paths.py`), so you can run it from any working directory.

### If the spec build misses imports

Heavy deps (e.g. `torch`, `cv2`, `onnxruntime`) are usually collected automatically. If you see import errors:

1. Add the missing module to **`hiddenimports`** in `dot.spec`.
2. Or use the **legacy manual method** below (bundles the full `site-packages`).

---

## Legacy: PyInstaller without spec (manual data)

If the spec file fails (e.g. missing DLLs or imports), use a one‑dir build and bundle `site-packages` yourself.

1. From repo root, activate the same environment you use to run dot.
2. Get site-packages path:
   ```bash
   python -c "import site; print(''.join(site.getsitepackages()))"
   ```
3. Run PyInstaller (replace `path/to/site-packages` with the path from step 2).

**Windows** (use `;` in `--add-data`):

```bash
pyinstaller --noconfirm --onedir --name "dot" ^
  --add-data "src/dot/fomm/config;dot/fomm/config" ^
  --add-data "src/dot/simswap/models;dot/simswap/models" ^
  --add-data "path/to/site-packages;." ^
  --add-data "configs;configs/" ^
  --add-data "data;data/" ^
  --add-data "saved_models;saved_models/" ^
  src/dot/ui/ui.py
```

**macOS / Linux** (use `:` in `--add-data`):

```bash
pyinstaller --noconfirm --onedir --name "dot" \
  --add-data "src/dot/fomm/config:dot/fomm/config" \
  --add-data "src/dot/simswap/models:dot/simswap/models" \
  --add-data "path/to/site-packages:." \
  --add-data "configs:configs/" \
  --add-data "data:data/" \
  --add-data "saved_models:saved_models/" \
  src/dot/ui/ui.py
```

Output is under `dist/dot/`.

---

## Ubuntu / Linux

Same as “Recommended” above: use `dot.spec` from repo root. If you need the legacy method, use the macOS/Linux `--add-data` form (with `:`). For a proper `.desktop` app you can wrap `dist/dot/dot` in a script or desktop file that sets `LD_LIBRARY_PATH` if needed.

---

## Nuitka (experimental)

For a single native binary you can try Nuitka. This is **not** the default path and can be fragile with PyTorch/CUDA/ONNX:

```bash
pip install nuitka
nuitka --standalone --onefile --enable-plugin=tk-inter --include-data-dir=configs=configs --include-data-dir=data=data src/dot/ui/ui.py
```

You may need to add more `--include-module` / `--include-data-dir` for torch and model dirs. Prefer PyInstaller unless you have a specific need for Nuitka.
