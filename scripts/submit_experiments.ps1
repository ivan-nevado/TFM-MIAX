<#
Submits experiments to AWS Batch, the same way as the replication run (array job, one child per
year, Spot): for each config a `train` array, then a `report` job that waits for it; finally one
`compare` job that waits for every report and writes reports/experiments (leaderboard + HISTORY).

Prerequisites (see docs/experiments/README.md, "Run on AWS"):
  aws sso login --profile relval ; $env:AWS_PROFILE = "relval"
  the training image with this code is pushed and the job definition points to it

Examples:
  # dry run: print the commands only
  ./scripts/submit_experiments.ps1 -Configs experiments/e1_l1.yaml,experiments/e1_huber.yaml -DryRun
  # E1 for real
  ./scripts/submit_experiments.ps1 -Configs experiments/e1_l1.yaml,experiments/e1_huber.yaml
  # quick check of one experiment on two years, no compare
  ./scripts/submit_experiments.ps1 -Configs experiments/e2_w12.yaml -FirstYear 2018 -LastYear 2019 -NoCompare
#>
param(
    [Parameter(Mandatory = $true)][string[]]$Configs,
    [int]$FirstYear = 1990,
    [int]$LastYear = 2019,
    [int]$Chunk = 1,
    [string]$Group = "base",
    [string]$Queue = "relval-dev-training",
    [string]$JobDefinition = "relval-dev-pipeline",
    [switch]$NoReport,
    [switch]$NoCompare,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"

# accept "a.yaml,b.yaml" passed as one string as well as an array
$Configs = $Configs | ForEach-Object { $_ -split "," } | Where-Object { $_ } | ForEach-Object { $_.Trim() }

function Submit-Job([string[]]$BatchArgs) {
    if ($DryRun) {
        Write-Host "aws batch submit-job $($BatchArgs -join ' ')"
        return "dry-run-$([guid]::NewGuid().ToString().Substring(0, 8))"
    }
    $out = & aws batch submit-job @BatchArgs --query jobId --output text
    if ($LASTEXITCODE -ne 0) { throw "submit-job failed: $($BatchArgs -join ' ')" }
    return $out.Trim()
}

$years = $LastYear - $FirstYear + 1
$children = [math]::Ceiling($years / $Chunk)
$reportIds = @()

foreach ($config in $Configs) {
    if (-not (Test-Path (Join-Path "ml/configs" $config))) { throw "config not found: ml/configs/$config" }
    $id = [IO.Path]::GetFileNameWithoutExtension($config)
    $params = "stage=train,group=$Group,first_year=$FirstYear,chunk=$Chunk,config=$config"
    $trainArgs = @("--job-name", "exp-$id-train", "--job-queue", $Queue, "--job-definition", $JobDefinition,
                   "--parameters", $params)
    if ($children -ge 2) { $trainArgs += @("--array-properties", "size=$children") }  # Batch arrays need >= 2
    $trainId = Submit-Job $trainArgs
    Write-Host "[$id] train: $trainId ($children children, $FirstYear-$LastYear)"

    if (-not $NoReport) {
        $reportId = Submit-Job @("--job-name", "exp-$id-report", "--job-queue", $Queue,
            "--job-definition", $JobDefinition, "--depends-on", "jobId=$trainId",
            "--parameters", "stage=report,group=$Group,first_year=$FirstYear,chunk=1,config=$config")
        Write-Host "[$id] report: $reportId (after train)"
        $reportIds += $reportId
    } else {
        $reportIds += $trainId
    }
}

if (-not $NoCompare) {
    $compareArgs = @("--job-name", "exp-compare", "--job-queue", $Queue, "--job-definition", $JobDefinition,
                     "--parameters", "stage=compare,group=$Group,first_year=$FirstYear,chunk=1,config=replication.yaml")
    # one --depends-on followed by every dependency (repeating the flag keeps only the last one)
    $compareArgs += @("--depends-on") + ($reportIds | ForEach-Object { "jobId=$_" })
    $compareId = Submit-Job $compareArgs
    Write-Host "compare: $compareId (after every report)"
}

Write-Host ""
Write-Host "Follow:  aws batch list-jobs --job-queue $Queue --filters name=JOB_NAME,values='exp-*' --query 'jobSummaryList[].[jobName,status,arrayProperties.statusSummary]'"
Write-Host "Logs:    aws logs tail /relval-dev/training --follow"
Write-Host "Results: aws s3 sync s3://<artifacts-bucket>/reports/experiments reports/experiments"
