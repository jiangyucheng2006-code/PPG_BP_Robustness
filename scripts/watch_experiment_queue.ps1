$ErrorActionPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $projectRoot "outputs"
$queues = @(
    @{ Name = "B1/B2 queue"; Path = (Join-Path $outputRoot "b1_b2_queue\status.json") },
    @{ Name = "B2 controlled queue"; Path = (Join-Path $outputRoot "b2_control_queue\status.json") },
    @{ Name = "B3 attention queue"; Path = (Join-Path $outputRoot "b3_attention_queue\status.json") }
)
$runs = @(
    @{ Name = "B1"; Directory = "pulsedb_b1_huber_cosine" },
    @{ Name = "B2"; Directory = "pulsedb_b2_multiscale" },
    @{ Name = "B2 controlled"; Directory = "pulsedb_b2_multiscale_control" },
    @{ Name = "B3 attention"; Directory = "pulsedb_b3_attention_multiscale_control" }
)

while ($true) {
    Clear-Host
    Write-Host "PPG BP experiment progress" -ForegroundColor Cyan
    Write-Host ("Updated: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ""

    foreach ($queueInfo in $queues) {
        if (Test-Path $queueInfo.Path) {
            $queue = Get-Content $queueInfo.Path -Raw | ConvertFrom-Json
            Write-Host ($queueInfo.Name + ": " + $queue.state)
        }
    }

    foreach ($run in $runs) {
        $directory = Join-Path $outputRoot $run.Directory
        $metricsPath = Join-Path $directory "metrics.json"
        $historyPath = Join-Path $directory "history.json"
        $logPath = Join-Path $directory "train_stderr.log"

        Write-Host ""
        Write-Host ($run.Name + ":") -ForegroundColor Yellow
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
    Write-Host "Refreshes every 5 seconds. Press Ctrl+C or close this window to stop viewing."
    Start-Sleep -Seconds 5
}
