# Temporal metric regression tests

From the repository root, using Python 3.12 and the existing metric dependencies
(`torch==2.4.0`, `nncore==0.4.5`):

```shell
python -m unittest discover -s tests -v
```

CPU PyTorch is sufficient. No models, datasets, API keys, caption-scoring
packages, or GPU are needed. The tests extract and execute the actual
`tvg_format`, `evs_eval`, and `tal_eval` function definitions from all four metric
files. This avoids the unrelated heavy imports and top-level conversion code;
it is not a full CLI/inference integration test. Tensor operations and temporal
IoU use real PyTorch and NNCore, without dependency stubs.

Expected values are checked against both independent Python set/interval
calculations and the repository's unchanged original ETBench scorer
(`all_benchmark_eval/etbench/compute_metrics_original_etchat.py`). The original
single-channel scorer receives answer-only or evidence-only text. The tests
also preserve the existing missing-evidence fallback and failed-answer
denominators, and check that later misses cannot erase earlier TAL scores.
