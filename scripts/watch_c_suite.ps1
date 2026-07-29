$ErrorActionPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $projectRoot "outputs"
$statusPath = Join-Path $outputRoot "c_robustness_suite\status.json"
$runs = @(
    @{ Name = "C0  baseline robustness"; Directory = "pulsedb_c0_baseline_robustness" },
    @{ Name = "C1  weak artifacts"; Directory = "pulsedb_c1_weak_artifacts" },
    @{ Name = "C2  full artifacts"; Directory = "pulsedb_c2_full_artifacts" },
    @{ Name = "C3  contact + motion"; Directory = "pulsedb_c3_contact_motion" },
    @{ Name = "C4  consistency"; Directory = "pulsedb_c4_consistency" },
    @{ Name = "C5  artifact-aware"; Directory = "pulsedb_c5_artifact_aware" },
    @{ Name = "C6a STP pretraining"; Directory = "pulsedb_c6_stp_pretrain" },
    @{ Name = "C6b STP fine-tuning"; Directory = "pulsedb_c6_stp_finetune" }
)

while ($true) {
    Clear-Host
    Write-Host "C-series PPG robustness progress" -ForegroundColor Cyan
    Write-Host ("Updated: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    if (Test-Path $statusPath) {
        $status = Get-Content $statusPath -Raw | ConvertFrom-Json
        Write-Host ("Suite: " + $status.state)
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
            if ($metrics.clean_test) {
                Write-Host (
                    "  Completed | clean test MAE={0:N3}" -f
                    $metrics.clean_test.mean_mae
                )
            } elseif ($metrics.best_validation_accuracy) {
                Write-Host (
                    "  Completed | best transform accuracy={0:P1}" -f
                    $metrics.best_validation_accuracy
                )
            } else {
                Write-Host "  Completed"
            }
        } elseif (Test-Path $historyPath) {
            $history = @(Get-Content $historyPath -Raw | ConvertFrom-Json)
            $last = $history[-1]
            if ($last.selection_score) {
                Write-Host (
                    "  Epochs={0} | clean val={1:N3} | robust val={2:N3} | selection={3:N3}" -f
                    $history.Count,
                    $last.mean_mae,
                    $last.robust_mean_mae,
                    $last.selection_score
                )
            } elseif ($last.val_accuracy) {
                Write-Host (
                    "  Pretrain epochs={0} | validation accuracy={1:P1}" -f
                    $history.Count,
                    $last.val_accuracy
                )
            }
            if (Test-Path $logPath) {
                $progress = Get-Content $logPath -Tail 20 |
                    Where-Object { $_ -match "(Epoch|Pretrain epoch) \d+:" } |
                    Select-Object -Last 1
                if ($progress) {
                    Write-Host ("  " + $progress)
                }
            }
        } elseif (Test-Path $logPath) {
            Write-Host "  Starting..."
            Get-Content $logPath -Tail 3 | ForEach-Object {
                Write-Host ("  " + $_)
            }
        } else {
            Write-Host "  Waiting"
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
    Write-Host "Refreshes every 5 seconds. Closing this viewer does not stop training."
    Start-Sleep -Seconds 5
}
