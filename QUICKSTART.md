# Quick Start

## Install

```bash
pip install -r requirements.txt    # numpy, tqdm, rustworkx
```

## Run (Enhanced — SHAP + Rip-up & Reroute)

```bash
# Quick test (10k nets, ~5 min)
python router.py -cap test_data/ariane.cap -net test_data/ariane.net \
    -output results.txt -enhanced -max_nets 10000

# Full design (105k nets, ~30 min)
python router.py -cap test_data/ariane.cap -net test_data/ariane.net \
    -output results.txt -enhanced -rr_iters 3 -rr_pct 20
```

## Run (Baseline — Steiner only)

```bash
python router.py -cap test_data/ariane.cap -net test_data/ariane.net \
    -output results.txt -batch 10000
```

## Test

```bash
python test_simple.py
```

## What You Get

The enhanced router prints:
1. Per-iteration SHAP analysis (layer importance, hotspot maps)
2. R&R progress (overflow reduction per iteration)
3. Full ISPD 2025 scoring table

```
Overflow (L2): 741,791  vs SOTA 4,349,152  (5.9x better)
S_orig: -0.009  vs SOTA 1.780
```

See [Results.md](Results.md) and [Novelty.md](Novelty.md) for details.
