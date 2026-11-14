# storage

Three private S3 buckets: `raw` (the licensed LSEG download), `processed` (panel) and `artifacts`
(predictions, models, SHAP values, peer weights, reports). Each is versioned, encrypted, blocks
all public access, refuses non-TLS requests and expires old object versions.

The licensed data may only ever go to the `raw` bucket of the owner's account, and only when the
owner uploads it. Nothing in this repository uploads data.

Outputs: `bucket_names`, `bucket_arns` (maps keyed by `raw`, `processed`, `artifacts`).
