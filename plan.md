# ReActor Quality Improvement Plan

Improvements to the ReActor face-swap pipeline for more robust KYC stress testing.

---

# Part A: Model Upgrades

## 0a. Switch to inswapper_128.fp16.onnx (Free Lunch)

**Problem:** Current `inswapper_128.onnx` is fp32 (~554 MB). An fp16 variant exists that is a byte-for-byte drop-in replacement with half the VRAM and faster tensor-core inference.

**Location:** `src/dot/reactor/option.py` — `INSWAPPER_REPO` / `INSWAPPER_FILE` constants

**Change:**
- Host or reference `inswapper_128.fp16.onnx` (~277 MB) from HuggingFace (`netrunner-exe/Insight-Swap-models`)
- Update the constants to point to the fp16 variant
- Add a config option to choose fp16 vs fp32

**Quality:** Identical. Negligible numerical difference.

**Effort:** Trivial (constant change + config flag)

---

## 0b. Add HyperSwap 256 Support (Biggest Quality Jump)

**Problem:** inswapper_128 operates at 128x128 native resolution. HyperSwap (FaceFusion Labs, 2025) runs at 256x256 — 4x the pixel budget — and is FaceFusion's new default model as of v3.3.

**Location:** New model integration in `src/dot/reactor/option.py` or a new `src/dot/hyperswap/` module

**Approach:**
- HyperSwap ships as ONNX with three variants: 1a (fast), 1b (balanced), 1c (quality)
- Same API pattern as inswapper: takes source face + target image, returns swapped result
- Integrate via `insightface.model_zoo.get_model()` or direct ONNX runtime session
- Add `swap_model` config option: `inswapper_128` | `inswapper_128_fp16` | `hyperswap_256`
- HyperSwap 256 produces dramatically sharper output that needs less face restoration

**Quality:** Major improvement. Native 256x256 means CodeFormer/GPEN have 4x more detail to work with.

**License:** ResearchRAIL-MS (non-commercial default; contact FaceFusion Labs for commercial)

**Effort:** Medium (~80 lines for new model loader + config)

---

## 0c. Add Ghost 256 / BlendSwap 256 as Alternatives

**Problem:** Only one swap model is available. Different models have different strengths.

**Models:**
| Model | Size | License | Strength |
|-------|------|---------|----------|
| BlendSwap 256 | 93 MB | Research | Best identity fidelity per compute |
| Ghost 1 (256) | 340 MB | Apache-2.0 | Fast, commercially friendly |
| Ghost 3 (256) | 739 MB | Apache-2.0 | Highest quality Ghost variant |

**Approach:**
- All are ONNX models with similar input/output signatures
- Add a model registry pattern so `swap_model` config selects which ONNX model to load
- Download from HuggingFace on first use (same as current inswapper download pattern)

**Effort:** Medium (~60 lines for model registry + download logic)

---

## 0d. ReSwapper 256 (Trainable Open-Source Option)

**Problem:** All InsightFace models are closed-source weights. If you need a custom-trained model (e.g., optimized for specific demographics or lighting conditions in your KYC test targets), there's no path.

**What it is:** ReSwapper is a fully open reproduction of inswapper with published training code + weights. Available at 128 and 256 resolution. Uses the same INSwapper class interface — true drop-in.

**Approach:**
- Add `reswapper_256` as a model option
- ONNX weights available on HuggingFace (`somanchiu/reswapper`)
- Same `model_zoo.get_model()` loading path
- Optionally: fine-tune on your own data for specific KYC scenarios

**License:** AGPL-3.0

**Effort:** Low (same integration as inswapper, just different weights)

---

# Part B: Pipeline Improvements

## 1. Color Correction (LAB Histogram Matching)

**Problem:** Swapped face retains source image lighting/skin tone, creating a visible mismatch with the target frame's neck, ears, and background.

**Location:** `src/dot/reactor/option.py` — `process_image()`

**Approach:**
- After `swapper.get()` returns the raw result, extract the swapped face region using the target bbox
- Convert both the swapped face region and original target face region to LAB color space
- Match the L (lightness) and A/B (color) channels of the swapped face to the target's statistics (mean + std transfer)
- Blend the color-corrected face back into the result

**Effort:** Low (~30 lines)

---

## 2. Poisson Blending (seamlessClone)

**Problem:** Current blending uses an elliptical mask with fixed Gaussian blur (`_mask_from_bbox`), producing visible halo artifacts around jaw and hairline.

**Location:** `src/dot/reactor/option.py` — `process_image()`

**Approach:**
- After the swap, compute a center point from the target face bbox
- Create a tight mask from the face region
- Use `cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)` to blend the swapped face into the original frame
- This eliminates hard boundaries by solving a Poisson equation across the seam

**Effort:** Low (~20 lines)

---

## 3. Enable GPEN 512 for ReActor

**Problem:** GPEN face super-resolution is wired up in `post_process_image()` but not activated in the reactor config.

**Location:** `configs/reactor.yaml`

**Change:**
```yaml
gpen_type: gpen_512
gpen_path: saved_models/gpen
```

**Effort:** Trivial (config change)

---

## 4. CodeFormer Fidelity Weight

**Problem:** CodeFormer face restoration is used but the fidelity weight isn't exposed. Default trades off too much identity for quality (or vice versa).

**Location:** `src/dot/reactor/option.py` — face restorer initialization

**Approach:**
- Add `face_restore_fidelity` parameter (default 0.7) to `ReActorOption.__init__`
- Pass it through to CodeFormer when initializing the restorer
- 0.0 = max quality, 1.0 = max identity preservation; 0.7 is the sweet spot for KYC

**Effort:** Low (~10 lines)

---

## 5. Multi-Reference Face Embedding

**Problem:** `change_option()` uses a single source image. Identity preservation degrades with pose/lighting variation.

**Location:** `src/dot/reactor/option.py` — `change_option()`

**Approach:**
- Accept a list of source images (or a directory)
- Extract face embeddings from each
- Average the 512-d embedding vectors to produce a more robust identity representation
- Store the averaged embedding as `self.source_face`
- 3-5 reference images with varied angles and lighting is optimal

**Effort:** Medium (~40 lines)

---

## 6. Face Parsing Mask (BiSeNet)

**Problem:** Elliptical bbox mask doesn't follow actual face boundaries. Includes background, misses chin, clips hair.

**Location:** `src/dot/reactor/option.py` — `process_image()`, replaces `_mask_from_bbox` usage

**Approach:**
- Add BiSeNet face parsing model (lightweight, <5ms per frame)
- Generate pixel-accurate mask for skin + face regions (exclude hair, background, ears as needed)
- Use this mask for both blending and color correction region selection
- Fall back to elliptical mask if parsing fails

**Effort:** Medium (~60 lines + model download)

---

## 7. Face Analyzer det_size

**Problem:** `FaceAnalysis.prepare()` uses default det_size (640x640). Higher resolution inputs benefit from larger detection size.

**Location:** `src/dot/diffusion/diffusion_option.py` — `face_analyzer` property

**Change:**
```python
self._face_analyzer.prepare(ctx_id=0 if self.use_gpu else -1, det_size=(1024, 1024))
```

**Effort:** Trivial (1 line)

---

## Implementation Order

```
Phase 1 — Quick wins (no code / trivial):
  [0a] Switch to inswapper_128.fp16    — constant change, free perf win
  [3]  Enable GPEN 512                 — config only
  [7]  det_size tuning                 — 1 line

Phase 2 — High-impact pipeline fixes:
  [1]  Color correction                — ~30 lines in process_image()
  [2]  Poisson blending                — ~20 lines in process_image()

Phase 3 — Model upgrade (biggest quality jump):
  [0b] HyperSwap 256 integration       — ~80 lines, new model loader
  [0c] Ghost/BlendSwap alternatives     — ~60 lines, model registry

Phase 4 — Refinement:
  [4]  CodeFormer fidelity             — ~10 lines, new param
  [5]  Multi-reference embed           — ~40 lines in change_option()
  [0d] ReSwapper 256 (trainable)       — same integration, different weights

Phase 5 — Advanced:
  [6]  Face parsing mask               — ~60 lines + BiSeNet model
```
