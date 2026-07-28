$ErrorActionPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $projectRoot "outputs"
$statusPath = Join-Path $outputRoot "multitask_stp_queue\status.json"
$runs = @(
    @{ Name = "M0  BP only"; Directory = "pulsedb_m0_bp_only" },
    @{ Name = "M1  + heart rate"; Directory = "pulsedb_m1_heart_rate" },
    @{ Name = "M2  + age group"; Directory = "pulsedb_m2_age_group" },
    @{ Name = "M3  + HR + age"; Directory = "pulsedb_m3_hr_age" },
    @{ Name = "M4  + HR + age + BP class"; Directory = "pulsedb_m4_hr_age_bpclass" }
)

while ($true) {
    Clear-Host
    Write-Host "M0-M4 multi-task experiment progress" -ForegroundColor Cyan
    Write-Host ("Updated: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    if (Test-Path $statusPath) {
        $queue = Get-Content $statusPath -Raw | ConvertFrom-Json
        Write-Host ("Queue: " + $queue.state)
    }

    foreach ($run in $runs) {
        $directory = Join-Path $outputRoot $run.Directory
        $metricsPath = Join-Path $directory "metrics.json"
        $historyPath = Join-Path $directory "history.json"
        $logPath = Join-Path $directory "train_stderr.log"
        Write-Host ""
        Write-Host $run.Name -ForegroundColor Yellow
        if (Test-Path $metricsPath) {
            $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
            Write-Host (
                "  Completed | epochs={0} | best epoch={1} | test MAE={2:N3}" -f
                $metrics.completed_epochs,
                $metrics.best_epoch,
                $metrics.test.mean_mae
            )
        } elseif (Test-Path $historyPath) {
            $history = @(Get-Content $historyPath -Raw | ConvertFrom-Json)
            $last = $history[-1]
            $best = $history | Sort-Object mean_mae | Select-Object -First 1
            Write-Host (
                "  Finished epochs={0} | last val MAE={1:N3} | best={2:N3} (epoch {3})" -f
                $history.Count,
                $last.mean_mae,
                $best.mean_mae,
                $best.epoch
            )
            if (Test-Path $logPath) {
                $progress = Get-Content $logPath -Tail 20 |
                    Where-Object { $_ -match "Epoch \d+:" } |
                    Select-Object -Last 1
                if ($progress) {
                    Write-Host ("  " + $progress)
                }
            }
        } elseif (Test-Path $logPath) {
            Write-Host "  Starting first epoch..."
            $progress = Get-Content $logPath -Tail 20 |
                Where-Object { $_ -match "Epoch \d+:" } |
                Select-Object -Last 1
            if ($progress) {
                Write-Host ("  " + $progress)
            }
        } else {
            Write-Host "  Waiting to start"
        }
    }

    Write-Host ""
    $gpu = & nvidia-smi `
        --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
        --format=csv,noheader,nounits 2>$null
    if ($gpu) {
        $values = $gpu -split ","
        Write-Host (
            "GPU: {0}% | VRAM: {1}/{2} MiB | Temperature: {3} C" -f
            $values[0].Trim(),
            $values[1].Trim(),
            $values[2].Trim(),
            $values[3].Trim()
        )
    }
    Write-Host ""
    Write-Host "Refreshes every 5 seconds. Press Ctrl+C only to stop this viewer."
    Start-Sleep -Seconds 5
}
