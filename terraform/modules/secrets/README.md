# secrets

Declares the names of the configuration parameters (SSM Parameter Store) and secrets (Secrets
Manager) of an environment. Values never appear in the code or in the state on purpose:
parameters start as `CHANGE_ME` and later changes are ignored; secrets are created without a
version. The owner sets the values in the console or with the AWS CLI.

Outputs: `parameter_arns`, `secret_arns`.
