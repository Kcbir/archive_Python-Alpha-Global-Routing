# Benchmark Logs Directory

This directory contains comprehensive logs from the enhanced router v2 benchmarks, demonstrating the integration of SHAP analysis, LLM guidance, Knowledge Graph recording, and Dijkstra maze routing.

## Log Files

### `v2_10k_nets_enhanced_router.log` (337KB)
- **Benchmark**: 10,000 nets from Ariane design
- **R&R Iterations**: 3
- **Final Result**: 133,748 overflow (32.5x better than ISPD 2025 SOTA)
- **Runtime**: ~824 seconds
- **Features Demonstrated**:
  - Flexible L-routing (dual L-shape evaluation)
  - Dijkstra maze routing for medium nets
  - LLM policy adaptation (Llama 3.2 3B)
  - SHAP-driven rip-up candidate selection
  - Knowledge Graph pattern recording

### `v2_105k_nets_enhanced_router.log` (754KB)
- **Benchmark**: Full 105,924 nets from Ariane design
- **R&R Iterations**: 5
- **Final Result**: 191,587 overflow (22.7x better than ISPD 2025 SOTA)
- **Runtime**: ~2,588 seconds
- **Features Demonstrated**:
  - All v2 features at scale
  - LLM guidance across 5 iterations
  - Progressive policy adaptation
  - Large-scale Knowledge Graph recording

## Log Content Analysis

Both logs contain detailed traces of:

### SHAP Feature Importance Reports
```
Layer-wise overflow decomposition:
  Layer 5 (V): 44.4% of total overflow
  Layer 2 (H): 25.6% of total overflow
  Util-overflow correlation: 0.2478
```

### LLM Policy Adaptation Reasoning
```
[LLM] llm: ['High overflow (614,713) indicates congestion; increasing alpha
will strengthen penalty for congested cells.', 'Low util-overflow correlation
(0.304) suggests that current parameter values may not be optimal...']
```

### Knowledge Graph Pattern Recording
```
KG Pattern: area=500-1000, pins=3-5, layer_pct=0.3-0.4
Strategy: alpha=0.53, pf=2.88, algo=maze
Outcome: overflow_delta=-0.67
```

### Dijkstra Maze Routing Decisions
```
Maze routing for net #12345 (area=8500, bbox=45x38)
Path found: 12 segments, detour around congested region
```

### ISPD 2025 Scoring Metrics
```
S_orig: -0.1737
T (runtime factor): 0.1418
S_scaled: -0.1490
```

## Usage

These logs serve as:
1. **Reproducibility evidence** for the enhanced router
2. **Debugging traces** for understanding routing decisions
3. **Research data** for analyzing LLM guidance effectiveness
4. **Performance benchmarks** for comparing with future improvements

## File Format

- Plain text logs with structured output
- Progress bars from tqdm (may appear as multiple lines)
- JSON-like structures for LLM responses
- Tabular data for SHAP reports
- ISPD 2025 standard scoring format

## Related Files

- `router.py`: The enhanced router implementation
- `Results.md`: Summary of benchmark results
- `Novelty.md`: Technical novelty claims
- `TECHNICAL_VALIDATION.md`: Analysis of active vs infrastructure components