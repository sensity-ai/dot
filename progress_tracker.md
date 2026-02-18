# Where We're Up To

## Status: Back to baseline, ready to start fresh

All code changes have been reverted. The codebase is clean (same as git HEAD).

## What we tried (Round 1 — all reverted)
- fp16 model swap
- GPEN 512 enable
- Color correction (LAB histogram matching)
- Poisson blending (seamlessClone)
- CodeFormer face restoration
- Configurable det_size
- New CLI args for all the above

## Why it was reverted
- Stacking everything at once made it slow and ugly
- GPEN + CodeFormer ran simultaneously (redundant, double neural net per frame)
- Poisson blending + CodeFormer defaulted to ON (killed camera performance)
- Without any enhancement the raw 128px swap output looked terrible

## Key lessons
1. Test ONE change at a time
2. Camera/video needs speed, single-image can afford expensive ops
3. GPEN and CodeFormer are mutually exclusive — never both
4. The real quality bottleneck is 128px native resolution (HyperSwap 256 would be the big jump)

## Next step when you're ready
- Start with Experiment 1: fp16 model (just 2 constants in option.py)
- See EXPERIMENTS.md for the full tracking sheet
- See plan.md for the full improvement plan

## Useful commands
- Camera swap: python -m dot --swap_type reactor --source ./data/jenna.jpg --target 1 --virtual_cam --use_gpu
- GUI with sliders: python -m dot --gui
