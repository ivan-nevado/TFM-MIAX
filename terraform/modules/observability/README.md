# observability

CloudWatch log groups for the training jobs and the API, an SNS topic for notifications, alarms
(API server errors, API CPU, failed Batch jobs) and a monthly cost budget filtered by the
`Project` tag, with notifications at 80% of actual and 100% of forecast spend.

Outputs: `log_group_names`, `log_group_arns`, `alarm_topic_arn`.
