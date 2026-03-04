# PAGR: Python Alpha Global Routing

SHAP-guided adaptive global router for ISPD 2025 performance-driven routing.

**22.7x lower overflow than ISPD 2025 SOTA** on Ariane benchmark (192K vs 4.35M, full 105k-net design with Dijkstra maze + KG + LLM).

## Architecture

```
.cap + .net --> Slack-Aware Ordering --> Steiner/MST/Dijkstra Routing --> SHAP Analysis
                                              ^                               |
                                              |                               v
                                        Rip-up & Reroute       Knowledge Graph (record)
                                        (Dijkstra maze +            |
                                         inflated costs)            v
                                              ^             LLM Log Analyzer
                                              |             (Ollama llama3.2)
                                              |                    |
                                              +--- Policy Adapter <+
                                                   (alpha, pf, via)
```

## Files

| File | Purpose |
|---|---|
| `router.py` | Router: Steiner + MST + Dijkstra + SHAP + KG + LLM + R&R (~1740 lines) |
| `utils.py` | `.cap` / `.net` parsers |
| `evaluate_solution.py` | Wrapper for C++ evaluator |
| `rl_simple_state.py` | RustWorkX graph state + PyG export (RL infra) |
| `rl_simple_agent.py` | GraphSAGE/CNN policy + REINFORCE (RL infra) |
| `test_simple.py` | Test suite |
| `Results.md` | Full benchmark results |
| `Novelty.md` | Technical novelty claims + SOTA comparison |

## Usage

```bash
# Install
pip install numpy tqdm rustworkx

# (Optional) Install Ollama for LLM analysis
brew install --cask ollama
ollama pull llama3.2:3b

# Baseline (Steiner only)
python router.py -cap test_data/ariane.cap -net test_data/ariane.net \
    -output results.txt -batch 10000

# Enhanced (SHAP + rip-up & reroute)
python router.py -cap test_data/ariane.cap -net test_data/ariane.net \
    -output results.txt -enhanced -rr_iters 3 -rr_pct 20

# Full enhanced with LLM + KG + Dijkstra maze
python router.py -cap test_data/ariane.cap -net test_data/ariane.net \
    -output results.txt -enhanced -llm -rr_iters 5

# Quick test (10k nets, ~14 min)
python router.py -cap test_data/ariane.cap -net test_data/ariane.net \
    -output results.txt -enhanced -llm -max_nets 10000
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `-cap` | required | `.cap` file path |
| `-net` | required | `.net` file path |
| `-output` | required | Output routing file |
| `-batch` | 5000 | Nets per batch |
| `-max_nets` | -1 (all) | Max nets to route |
| `-enhanced` | off | Enable SHAP + KG + R&R |
| `-rr_iters` | 5 | R&R iterations |
| `-rr_pct` | 20 | % of overflow nets to rip up |
| `-llm` | off | Enable Ollama LLM log analysis |
| `-no_kg` | off | Disable Knowledge Graph |
| `-no_maze` | off | Disable Dijkstra maze routing |

## Results (Ariane, 105,924 nets)

### 10k Net Test (Enhanced v2)

| Stage | Overflow (L2) | Change |
|---|---|---|
| INIT | 352,493 | — |
| R&R-1 | 157,283 | -55.4% |
| R&R-2 | 138,261 | -60.8% |
| R&R-3 | 133,748 | -62.1% |

### vs ISPD 2025 SOTA

| Metric | SOTA | Ours (full) | Factor |
|---|---|---|---|
| Overflow (L2) | 4,349,152 | 191,587 | **22.7x better** |
| S_orig | 1.780 | -0.174 | **significantly better** |
| Runtime | 10 s | 2,588 s | 259x slower (Python) |

See [Results.md](Results.md) for full benchmark data and [Novelty.md](Novelty.md) for technical details.

## Citation

```
@article{solovyev2025pagr,
  title={PAGR: Accelerating Global Routing for VLSI Design Flow},
  author={Solovyev, Roman A and Mkrtchan, Ilya A and Telpukhov, Dmitry V
          and Shafeev, Ilya I and Romanov, Aleksandr Y and Stolbikov,
          Yevgeniy V and Stempkovsky, Alexander L},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```
