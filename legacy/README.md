# legacy

Reserved for the team's earlier scripts: `construir_97_variables.py` (panel builder),
`unir_lotes_lseg.py` (merge of the download batches), the LSEG download notebook and the Excel
dictionaries.

None of them was in the repository when the replication was built, so there is nothing to
preserve here yet. The package `relval` reimplements both steps from the paper's Table A.1 and
from the raw download batches:

- merge of the batches: `ml/src/relval/data/lseg.py` (reads the zips in place);
- panel builder: `ml/src/relval/panel/` and `ml/src/relval/features/`.

If the original scripts are recovered, put them here unchanged. The differences between their
reported sample sizes and ours are explained in `docs/replication/DEVIATIONS.md`.
