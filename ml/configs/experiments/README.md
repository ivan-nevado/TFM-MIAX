# Experiment configs

One file per experiment. Each extends `_common.yaml` (which extends `../replication.yaml`) and
changes one idea. Rules:

- file name = `experiment.id`; `run_name: experiments/<id>` (outputs never mix with the replication)
- write `experiment.hypothesis` before running; never edit a file that has run: create a new one
- files starting with `_` are shared bases, not experiments

Catalogue, protocol, reading rule and how to run on AWS Batch: `docs/experiments/README.md`.
Results of every run: `reports/experiments/HISTORY.md` and `reports/experiments/comparison/`.
