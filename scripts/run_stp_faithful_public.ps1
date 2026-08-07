param(
    [switch]$SkipDataPreparation,
    [string]$RunName = "stp_public_method_v4",
    [string]$ProcessedName = "stp_public_method_v4"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$dataRoot = "D:\Datasets\PPG_BP_Robustness"
$rawRoot = Join-Path $dataRoot "STP_Public_Raw"
$processedRoot = Join-Path $dataRoot ("processed\{0}" -f $ProcessedName)
$runRoot = Join-Path $projectRoot ("outputs\{0}" -f $RunName)
$logRoot = Join-Path $dataRoot ("logs\{0}" -f $RunName)
$env:PPG_BP_DATA_ROOT = $dataRoot
$env:STP_RUN_ROOT = $runRoot
$env:STP_DATA_ROOT = $processedRoot

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

if (-not $SkipDataPreparation) {
    python -u scripts\prepare_stp_public_data.py `
        --output $processedRoot `
        --raw-root $rawRoot `
        --download-unpaired `
        --reuse-prepared `
        --mimic-max-subjects 200 `
        --discover-mimic-records `
        --discover-mimic-unpaired `
        --mimic-unpaired-subjects 300 `
        --strict-stp-counts
    if ($LASTEXITCODE -ne 0) { throw "STP data preparation failed." }
} elseif (-not (Test-Path (Join-Path $processedRoot "manifest.json"))) {
    throw "Cannot skip data preparation because manifest.json is missing."
} else {
    Write-Output "Reusing completed STP dataset manifest; data preparation skipped."
}

python -u scripts\audit_stp_public_data.py `
    --root $processedRoot `
    --output (Join-Path $runRoot "data_audit.json") `
    --strict
if ($LASTEXITCODE -ne 0) { throw "STP public-data audit failed." }

$stages = @(
    @{ Name = "F1"; Config = "configs\stp_faithful_public_s1.yaml"; Output = (Join-Path $runRoot "F1") },
    @{ Name = "F2"; Config = "configs\stp_faithful_public_s2.yaml"; Output = (Join-Path $runRoot "F2") },
    @{ Name = "F3"; Config = "configs\stp_faithful_public_s3.yaml"; Output = (Join-Path $runRoot "F3") }
)

foreach ($stage in $stages) {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -u scripts\train_stp.py `
        --config $stage.Config `
        --output $stage.Output `
        --resume 2>&1 | Tee-Object -FilePath (Join-Path $logRoot ("stp_faithful_{0}.log" -f $stage.Name.ToLower()))
    $stageExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($stageExitCode -ne 0) { throw "$($stage.Name) failed with exit code $stageExitCode." }
}
