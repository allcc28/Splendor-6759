# Event-based Splendor Project (isolated)

This directory contains an isolated copy of the event-based experiment code so
it can be developed and run independently from the original `project/` score-based code.

Structure mirrors the original `project/` layout for quick testing:
- `configs/` - training configs
- `scripts/` - training and debug scripts
- `src/` - minimal utilities required (vectorizer, detector, reward)

Run the smoke-test after creating a Conda env (see repository-level README or use `requirements.txt`).
