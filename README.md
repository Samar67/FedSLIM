# FedSLIM

Federated pattern mining using the SLIM algorithm. A central server and a set of clients collaboratively discover an MDL-optimal codetable from distributed transaction data without exchanging raw records. Two communication variants are provided, together with an evaluation suite for measuring result quality.

## Repository structure

### `FedSLIM_SA/` — Secure Aggregation variant

Implements FedSLIM with a two-round cryptographic secure aggregation protocol (ECDH key exchange + PRG masking) so the server only ever observes the *sum* of client usage vectors, never individual contributions. Entry point:

```
python run.py --dataset <name> --num_clients <N> --folder_name <data_dir>
```

Client transaction files are expected at `./data/<folder_name>/cl<id>.dat`. Mutual-TLS certificates are generated automatically under `./certs/`.

### `FedSLIM_SO/` — Exact usage communication variant

Implements FedSLIM where clients send their local usage counts directly to the server for aggregation. This variant serves as a fidelity upper bound and a baseline for comparing against the secure aggregation overhead. Entry point:

```
python run.py --dataset <name> --clients_num <N> --folder_name <data_dir>
```

### `Evaluation/` — Fidelity and gap metrics

Contains `run_all_metrics.py`, a self-contained script that computes two families of quality measures by comparing centralised, local, and federated codetable output files:

- **Fidelity** (federated output B vs. centralised output A): Recall, Precision, F1, weighted recall at top-50/100 patterns (WR@50, WR@100), and Spearman rank correlation (ρ) over matched itemset usage counts.
- **Discovery-gap metrics** (local-vs-global): gap size |G| = |A \ ∪CT_i|, target gap |G_target|, gap recovery rate GRR = |G ∩ B| / |G|, and spread factor s(X) = global_usage(X) / max_client_usage(X) for each gap itemset.

Run from the `Evaluation/` directory:

```
python run_all_metrics.py
```

Results are printed to stdout and saved to `Evaluation/all_metrics_results.txt`.
