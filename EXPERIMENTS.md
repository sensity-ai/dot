# ReActor Quality Experiments

Tracking each change individually: what was tried, speed impact, quality impact.

Test command (camera/virtual cam):
```bash
python -m dot --swap_type reactor --source ./data/sayang.jpg --target 1 --virtual_cam --use_gpu
```

---

## Baseline

- **Model:** inswapper_128.onnx (fp32, 554 MB)
- **Pipeline:** swap -> BGR convert -> return (no post-processing)
- **Config:** `configs/reactor.yaml` (no gpen_type, use_face_restore=true but CodeFormer not implemented)
- **Speed:** baseline reference
- **Quality:** baseline reference — 128x128 native swap, no enhancement, no color matching

---

## Experiment 1: Switch to fp16 model

- **Change:** Point to `inswapper_128.fp16.onnx` (277 MB) from `netrunner-exe/Insight-Swap-models`
- **Files:** `src/dot/reactor/option.py` (constants only)
- **Speed:** EXPECTED faster (half model size, tensor core friendly)
- **Quality:** EXPECTED identical (negligible numerical difference)
- **Actual speed:** _pending_
- **Actual quality:** _pending_
- **Verdict:** _pending_

---

## Experiment 2: Enable GPEN 512 via config

- **Change:** Add `gpen_type: gpen_512` to `configs/reactor.yaml`
- **Files:** `configs/reactor.yaml` only
- **Speed:** EXPECTED slower (~50-200ms/frame, neural net per frame)
- **Quality:** EXPECTED better (face super-resolution on 128px swap output)
- **Actual speed:** _pending_
- **Actual quality:** _pending_
- **Verdict:** _pending_

---

## Experiment 3: Color correction (LAB histogram matching)

- **Change:** After swap, match swapped face color stats to original target face in LAB space
- **Files:** `src/dot/reactor/option.py` — new `_color_correct()` method + call in `process_image()`
- **Speed:** EXPECTED negligible (<1ms, just numpy)
- **Quality:** EXPECTED better for stills, may cause flickering on camera
- **Actual speed:** _pending_
- **Actual quality:** _pending_
- **Verdict:** _pending_

---

## Experiment 4: Poisson blending (seamlessClone)

- **Change:** Use `cv2.seamlessClone` instead of raw paste_back to eliminate face boundary artifacts
- **Files:** `src/dot/reactor/option.py` — addition to `process_image()`
- **Speed:** EXPECTED slower (~30-100ms/frame, PDE solver)
- **Quality:** EXPECTED better (eliminates halo/boundary artifacts)
- **Actual speed:** _pending_
- **Actual quality:** _pending_
- **Verdict:** _pending_

---

## Experiment 5: Configurable det_size

- **Change:** Make `face_analyzer.prepare(det_size=...)` configurable
- **Files:** `src/dot/diffusion/diffusion_option.py`
- **Speed:** Higher det_size = slower detection, lower = faster but misses small faces
- **Quality:** Higher det_size = better detection of small/distant faces
- **Actual speed:** _pending_
- **Actual quality:** _pending_
- **Verdict:** _pending_

---

## Experiment 6: CodeFormer face restoration

- **Change:** Implement the CodeFormer face restore that's advertised but not wired up
- **Files:** `src/dot/reactor/option.py`, `pyproject.toml`
- **Speed:** EXPECTED much slower (~200-500ms/frame, neural net + redundant face detection)
- **Quality:** EXPECTED better for stills (face detail restoration)
- **WARNING:** Do NOT combine with GPEN — they are redundant face enhancers
- **Actual speed:** _pending_
- **Actual quality:** _pending_
- **Verdict:** _pending_

---

## Lessons Learned (Round 1 — reverted)

1. **Never enable multiple face enhancers simultaneously.** GPEN + CodeFormer = 2 neural nets per frame for the same purpose. Must be mutually exclusive.
2. **Expensive ops must be opt-in, not opt-out.** Poisson blending and CodeFormer defaulting to ON killed camera/video performance.
3. **Camera/video and single-image have fundamentally different needs.** Camera needs speed (swap + maybe GPEN 256). Single image can afford CodeFormer + Poisson.
4. **Test each change in isolation.** Stacking all changes at once made it impossible to tell what helped and what hurt.
5. **The 128px inswapper output is the root quality bottleneck.** Post-processing can only polish a 128px face so much. The plan's mention of HyperSwap 256 / Ghost 256 / BlendSwap 256 would be the real quality jump.

---

## Recommended Implementation Order

Implement ONE at a time. Test. Record results. Then decide next step.

1. **fp16 model** — zero quality risk, pure upside
2. **GPEN 512 config** — known quantity, just a config toggle
3. **Color correction** — cheap, test on both camera and stills
4. **det_size config** — trivial, lets user tune
5. **Poisson blending** — stills only, too slow for camera
6. **CodeFormer** — stills only, mutually exclusive with GPEN
