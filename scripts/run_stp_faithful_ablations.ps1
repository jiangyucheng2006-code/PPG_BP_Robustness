param(
    [int]$WaitForProcessId = 0,
    [string]$BaselineRunName = "stp_public_method_v2"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = "D:\Datasets\PPG_BP_Robustness"
$logRoot = Join-Path $dataRoot "logs"
$baselineRoot = Join-Path $projectRoot ("outputs\{0}" -f $BaselineRunName)
$summaryRoot = Join-Path $projectRoot ("outputs\{0}_ablations" -f $BaselineRunName)
$env:PPG_BP_DATA_ROOT = $dataRoot
$env:STP_RUN_ROOT = $baselineRoot

New-Item -ItemType Directory -Force -Path $logRoot, $summaryRoot | Out-Null
Set-Location $projectRoot

if ($WaitForProcessId -gt 0) {
    Write-Output "Waiting for faithful F1-F3 pipeline (PID $WaitForProcessId)..."
    Wait-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
}

$required = @(
    (Join-Path $baselineRoot "F1\best.pt"),
    (Join-Path $baselineRoot "F2\best.pt"),
    (Join-Path $baselineRoot "F3\metrics.json")
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Faithful baseline did not finish successfully; missing $path"
    }
}

$experiments = @(
    @{ Name = "direct_s1_seed42"; Path = "F1_to_F3"; Seed = 42; Config = "configs\stp_faithful_ablation_direct_s1.yaml"; Output = (Join-Path $summaryRoot "direct_s1_seed42") },
    @{ Name = "scratch_seed42"; Path = "random_to_F3"; Seed = 42; Config = "configs\stp_faithful_ablation_scratch.yaml"; Output = (Join-Path $summaryRoot "scratch_seed42") },
    @{ Name = "full_seed73"; Path = "F1_to_F2_to_F3"; Seed = 73; Config = "configs\stp_faithful_public_s3.yaml"; Output = (Join-Path $summaryRoot "full_seed73") },
    @{ Name = "direct_s1_seed73"; Path = "F1_to_F3"; Seed = 73; Config = "configs\stp_faithful_ablation_direct_s1.yaml"; Output = (Join-Path $summaryRoot "direct_s1_seed73") },
    @{ Name = "scratch_seed73"; Path = "random_to_F3"; Seed = 73; Config = "configs\stp_faithful_ablation_scratch.yaml"; Output = (Join-Path $summaryRoot "scratch_seed73") },
    @{ Name = "full_seed137"; Path = "F1_to_F2_to_F3"; Seed = 137; Config = "configs\stp_faithful_public_s3.yaml"; Output = (Join-Path $summaryRoot "full_seed137") }
)

foreach ($experiment in $experiments) {
    Write-Output ""
    Write-Output ("===== {0} =====" -f $experiment.Name)
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -u scripts\train_stp.py `
        --config $experiment.Config `
        --output $experiment.Output `
        --seed $experiment.Seed `
        --resume 2>&1 | Tee-Object -FilePath (Join-Path $logRoot ("stp_faithful_ablation_{0}.log" -f $experiment.Name))
    $experimentExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($experimentExitCode -ne 0) {
        throw "$($experiment.Name) failed with exit code $experimentExitCode."
    }
}

$rows = @()
$baseline = Get-Content (Join-Path $baselineRoot "F3\metrics.json") | ConvertFrom-Json
$rows += [pscustomobject]@{
    experiment = "full_seed42"
    transfer_path = "F1_to_F2_to_F3"
    seed = 42
    sbp_mae = $baseline.test.sbp_mae
    dbp_mae = $baseline.test.dbp_mae
    mean_mae = $baseline.test.mean_mae
    sbp_error_std = $baseline.test.sbp_error_std
    dbp_error_std = $baseline.test.dbp_error_std
}

foreach ($experiment in $experiments) {
    $metrics = Get-Content (Join-Path $experiment.Output "metrics.json") | ConvertFrom-Json
    $rows += [pscustomobject]@{
        experiment = $experiment.Name
        transfer_path = $experiment.Path
        seed = $experiment.Seed
        sbp_mae = $metrics.test.sbp_mae
        dbp_mae = $metrics.test.dbp_mae
        mean_mae = $metrics.test.mean_mae
        sbp_error_std = $metrics.test.sbp_error_std
        dbp_error_std = $metrics.test.dbp_error_std
    }
}

$sorted = $rows | Sort-Object mean_mae
$sorted | Export-Csv (Join-Path $summaryRoot "summary.csv") -NoTypeInformation -Encoding UTF8
$sorted | ConvertTo-Json | Out-File (Join-Path $summaryRoot "summary.json") -Encoding UTF8
Write-Output ("Ablation suite complete. Summary: {0}" -f (Join-Path $summaryRoot "summary.csv"))
