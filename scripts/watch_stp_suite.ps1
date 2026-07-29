$ErrorActionPreference = "SilentlyContinue"

$dataRoot = "D:\Datasets\PPG_BP_Robustness"
$processedRoot = Join-Path $dataRoot "processed\stp_public"
$rawRoot = Join-Path $dataRoot "STP_Public_Raw"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Format-Stage {
    param(
        [string]$Name,
        [string]$OutputDirectory
    )
    $historyPath = Join-Path $OutputDirectory "history.json"
    $metricsPath = Join-Path $OutputDirectory "metrics.json"
    if (Test-Path $metricsPath) {
        $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
        return ("{0,-5} COMPLETE  best epoch {1}, epochs {2}" -f `
            $Name, $metrics.best_epoch, $metrics.completed_epochs)
    }
    if (Test-Path $historyPath) {
        $history = Get-Content $historyPath -Raw | ConvertFrom-Json
        if ($history.Count -gt 0) {
            $last = $history[-1]
            return ("{0,-5} TRAINING  epoch {1}" -f $Name, $last.epoch)
        }
    }
    return ("{0,-5} WAITING" -f $Name)
}

while ($true) {
    Clear-Host
    Write-Host "STP S-SERIES LIVE MONITOR" -ForegroundColor Cyan
    Write-Host ("Updated: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ""

    $preparation = Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -like "*prepare_stp_public_data.py*"
        }
    $preparationState = if ($preparation) { "RUNNING" } else { "STOPPED" }
    Write-Host ("DATA PREPARATION: {0}" -f $preparationState) -ForegroundColor Yellow

    $wesad = @(
        Get-ChildItem (Join-Path $processedRoot "subjects\wesad") `
            -Filter "*_signals.npy"
    ).Count
    $dalia = @(
        Get-ChildItem (Join-Path $processedRoot "subjects\ppgdalia") `
            -Filter "*_signals.npy"
    ).Count
    $mimic = @(
        Get-ChildItem (Join-Path $processedRoot "subjects\mimiciii") `
            -Filter "*_signals.npy"
    ).Count

    $daliaArchive = Join-Path $rawRoot "PPG-DaLiA.zip"
    $daliaPartial = Join-Path $rawRoot "PPG-DaLiA.zip.part"
    if (Test-Path $daliaArchive) {
        $daliaDownload = "download complete"
    }
    elseif (Test-Path $daliaPartial) {
        $size = (Get-Item $daliaPartial).Length / 1GB
        $daliaDownload = ("downloading {0:N2} GB" -f $size)
    }
    else {
        $daliaDownload = "waiting"
    }

    Write-Host ("  WESAD:      {0}/15 subjects prepared" -f $wesad)
    Write-Host ("  PPG-DaLiA:  {0}/15 subjects prepared; {1}" -f `
        $dalia, $daliaDownload)
    Write-Host ("  MIMIC-III:  {0}/500 record groups prepared" -f $mimic)
    Write-Host ""

    Write-Host "TRAINING" -ForegroundColor Yellow
    Write-Host (Format-Stage "S1" (Join-Path $projectRoot "outputs\stp_s1_public_pretrain"))
    Write-Host (Format-Stage "S2" (Join-Path $projectRoot "outputs\stp_s2_public_pattern"))
    Write-Host (Format-Stage "S3" (Join-Path $projectRoot "outputs\stp_s3_public_bp"))
    Write-Host ""
    Write-Host "S1: reconstruction pretraining"
    Write-Host "S2: BP-pattern adaptation"
    Write-Host "S3: SBP/DBP regression"
    Write-Host ""
    Write-Host "Press Ctrl+C to close this monitor."
    Start-Sleep -Seconds 3
}
