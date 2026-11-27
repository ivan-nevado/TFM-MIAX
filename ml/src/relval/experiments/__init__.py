"""Extensions of the replication (E1 robust loss, E2 training window, ...).

Each experiment is one YAML file in ``ml/configs/experiments/`` that extends ``replication.yaml``
and changes as few keys as possible. Its outputs live under the run name ``experiments/<id>``
(``data/predictions/experiments/<id>``, ``reports/experiments/<id>``), never mixed with the
replication's. ``relval compare`` sets every experiment against the replication baseline.
"""
