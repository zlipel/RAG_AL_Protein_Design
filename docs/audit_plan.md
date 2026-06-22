# Agent Scaffold Audit Plan

Goal: Determine whether the generated project correctly implements retrospective active learning without label leakage.

## Critical questions

1. Does the AL loop reveal labels only after candidate selection?
2. Does the surrogate train only on currently labeled labels?
3. Does retrieval use labels only from the current labeled set?
4. Does representation-level retrieval avoid self-label leakage?
5. Do acquisition functions ever receive hidden pool labels?
6. Are embeddings computed without using fitness labels?
7. Are selected indices, configs, and metrics saved per run?
8. Are random seeds controlled?

## Audit status

All 8 questions resolved ✅. See `docs/bugs.md` for findings and fix history.