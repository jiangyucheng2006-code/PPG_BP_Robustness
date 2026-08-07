param(
    [int]$PreparationProcessId = 0
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = "D:\Datasets\PPG_BP_Robustness"
$processedRoot = Join-Path $dataRoot "processed\stp_public"
$logRoot = Join-Path $dataRoot "logs"
$env:PPG_BP_DATA_ROOT = $dataRoot

if ($PreparationProcessId -gt 0) {
    Write-Host "Waiting for public-data preparation (PID $PreparationProcessId)..."
    Wait-Process -Id $PreparationProcessId
}

$manifest = Join-Path $processedRoot "manifest.json"
if (-not (Test-Path $manifest)) {
    throw "Data preparation did not create $manifest; training was not started."
}

$stages = @(
    @{
        Name = "S1"
        Config = "configs\stp_s1_public_pretrain.yaml"
        Output = "outputs\stp_s1_public_pretrain"
    },
    @{
        Name = "S2"
        Config = "configs\stp_s2_public_pattern.yaml"
        Output = "outputs\stp_s2_public_pattern"
    },
    @{
        Name = "S3"
        Config = "configs\stp_s3_public_bp.yaml"
        Output = "outputs\stp_s3_public_bp"
    }
)

foreach ($stage in $stages) {
    $stdout = Join-Path $logRoot ("stp_{0}.out.log" -f $stage.Name.ToLower())
    $stderr = Join-Path $logRoot ("stp_{0}.err.log" -f $stage.Name.ToLower())
    Write-Host ("Starting {0}..." -f $stage.Name)
    $process = Start-Process `
        -FilePath (Get-Command python).Source `
        -ArgumentList @(
            "-u",
            "scripts\train_stp.py",
            "--config", $stage.Config,
            "--output", $stage.Output,
            "--resume"
        ) `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "$($stage.Name) failed with exit code $($process.ExitCode)."
    }
}

Write-Host "S1-S3 training complete."
