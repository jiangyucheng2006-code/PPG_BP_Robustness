$ErrorActionPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $projectRoot "outputs"
$queues = @(
    @{ Name = "B1/B2 queue"; Path = (Join-Path $outputRoot "b1_b2_queue\status.json") },
    @{ Name = "B2 controlled queue"; Path = (Join-Path $outputRoot "b2_control_queue\status.json") },
    @{ Name = "B3 attention queue"; Path = (Join-Path $outputRoot "b3_attention_queue\status.json") },
    @{ Name = "B3-v2 task attention queue"; Path = (Join-Path $outputRoot "b3v2_attention_queue\status.json") },
    @{ Name = "B4 derivative queue"; Path = (Join-Path $outputRoot "b4_derivative_queue\status.json") },
    @{ Name = "B5 density-weighted queue"; Path = (Join-Path $outputRoot "b5_density_queue\status.json") },
    @{ Name = "B6 gated-fusion queue"; Path = (Join-Path $outputRoot "b6_gated_queue\status.json") },
    @{ Name = "B6 heads control"; Path = (Join-Path $outputRoot "b6_heads_control_queue\status.json") },
    @{ Name = "B6 multi-seed sweep"; Path = (Join-Path $outputRoot "b6_seed_sweep_queue\status.json") },
    @{ Name = "B6 concat-attention screen"; Path = (Join-Path $outputRoot "b6_concat_attention_queue\status.json") }
)
$runs = @(
    @{ Name = "B1"; Directory = "pulsedb_b1_huber_cosine" },
    @{ Name = "B2"; Directory = "pulsedb_b2_multiscale" },
    @{ Name = "B2 controlled"; Directory = "pulsedb_b2_multiscale_control" },
    @{ Name = "B3 attention"; Directory = "pulsedb_b3_attention_multiscale_control" },
    @{ Name = "B3-v2 task attention"; Directory = "pulsedb_b3v2_task_attention_control" },
    @{ Name = "B4 PPG+VPG"; Directory = "pulsedb_b4_vpg_control" },
    @{ Name = "B4 PPG+VPG+APG"; Directory = "pulsedb_b4_vpg_apg_control" },
    @{ Name = "B5-1 moderate balance"; Directory = "pulsedb_b5_density_moderate" },
    @{ Name = "B5-2 strong balance"; Directory = "pulsedb_b5_density_strong" },
    @{ Name = "B6-1 gated PPG/VPG"; Directory = "pulsedb_b6_gated_fusion" },
    @{ Name = "B6-2 gated + task heads"; Directory = "pulsedb_b6_gated_task_heads" },
    @{ Name = "B6-3 task heads only"; Directory = "pulsedb_b6_independent_heads_control" },
    @{ Name = "B5-1 seed 7"; Directory = "pulsedb_b5_density_moderate_seed7" },
    @{ Name = "B6-2 attention seed 7"; Directory = "pulsedb_b6_gated_task_heads_seed7" },
    @{ Name = "B6-3 heads seed 7"; Directory = "pulsedb_b6_independent_heads_control_seed7" },
    @{ Name = "B5-1 seed 2026"; Directory = "pulsedb_b5_density_moderate_seed2026" },
    @{ Name = "B6-2 attention seed 2026"; Directory = "pulsedb_b6_gated_task_heads_seed2026" },
    @{ Name = "B6-3 heads seed 2026"; Directory = "pulsedb_b6_independent_heads_control_seed2026" },
    @{ Name = "B6-4 concat attention"; Directory = "pulsedb_b6_concat_attention_task_heads" }
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
