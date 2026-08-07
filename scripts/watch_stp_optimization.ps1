$ErrorActionPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = Join-Path $projectRoot "outputs\stp_optimization_suite"
$statusPath = Join-Path $outputRoot "status.json"
$summaryPath = Join-Path $outputRoot "summary.json"

while ($true) {
    Clear-Host
    Write-Host "STP OPTIMIZATION - LIVE STATUS" -ForegroundColor Cyan
    Write-Host ("Updated: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ""

    if (-not (Test-Path $statusPath)) {
        Write-Host "Waiting for the optimization queue to create status.json..."
        Start-Sleep -Seconds 3
        continue
    }

    $status = Get-Content $statusPath -Raw | ConvertFrom-Json
    $started = [DateTimeOffset]::Parse($status.started_at)
    $elapsed = [DateTimeOffset]::Now - $started
    Write-Host ("Queue state:      {0}" -f $status.state) -ForegroundColor Yellow
    Write-Host (
        "Completed stages: {0}/{1}" -f `
            $status.completed_stages, $status.planned_stages
    )
    Write-Host (
        "Elapsed:          {0:00}:{1:00}:{2:00}" -f `
            [math]::Floor($elapsed.TotalHours), $elapsed.Minutes, $elapsed.Seconds
    )
    Write-Host ("Estimated total:  {0} hours" -f $status.estimated_runtime_hours)
    Write-Host ""

    if ($null -ne $status.current) {
        $current = $status.current
        Write-Host "CURRENT EXPERIMENT" -ForegroundColor Yellow
        Write-Host ("  Stage:     {0}" -f $current.stage)
        Write-Host ("  Variant:   {0}" -f $current.variant)
        Write-Host ("  Path:      {0}" -f $current.transfer_path)
        Write-Host ("  Seed:      {0}" -f $current.seed)
        Write-Host ("  Purpose:   {0}" -f $current.description)
        $historyPath = Join-Path $current.output "history.json"
        if (Test-Path $historyPath) {
            $history = Get-Content $historyPath -Raw | ConvertFrom-Json
            if ($history.Count -gt 0) {
                $last = $history[-1]
                Write-Host ("  Epoch:     {0}/120" -f $last.epoch)
                if ($current.stage -eq "S2") {
                    Write-Host (
                        "  Validation macro recall: {0:P2}" -f `
                            $last.pattern_macro_recall
                    )
                }
                else {
                    Write-Host (
                        "  Validation MAE: SBP {0:N2}, DBP {1:N2}, mean {2:N2}" -f `
                            $last.sbp_mae, $last.dbp_mae, $last.mean_mae
                    )
                }
            }
        }
        Write-Host ""
    }

    if (Test-Path $summaryPath) {
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        Write-Host "BEST COMPLETED S3 RESULTS" -ForegroundColor Yellow
        foreach ($path in "direct_s1", "via_s2") {
            $best = $summary.best_by_transfer_path.$path
            if ($null -ne $best) {
                Write-Host (
                    "  {0}: {1}, seed {2}, mean MAE {3:N2} " + `
                    "(SBP {4:N2}, DBP {5:N2})" -f `
                        $path,
                        $best.variant,
                        $best.seed,
                        $best.metrics.mean_mae,
                        $best.metrics.sbp_mae,
                        $best.metrics.dbp_mae
                )
            }
            else {
                Write-Host ("  {0}: waiting" -f $path)
            }
        }
        Write-Host ""
        Write-Host "RECENT COMPLETED STAGES" -ForegroundColor Yellow
        @($summary.records) |
            Where-Object { $_.state -eq "complete" } |
            Select-Object -Last 5 |
            ForEach-Object {
                if ($_.stage -eq "S2") {
                    Write-Host (
                        "  S2 {0} seed {1}: macro recall {2:P2}" -f `
                            $_.variant,
                            $_.seed,
                            $_.metrics.pattern_macro_recall
                    )
                }
                else {
                    Write-Host (
                        "  S3 {0}/{1} seed {2}: mean MAE {3:N2}" -f `
                            $_.variant,
                            $_.transfer_path,
                            $_.seed,
                            $_.metrics.mean_mae
                    )
                }
            }
    }

    Write-Host ""
    Write-Host "Ctrl+C closes this monitor only; training continues."
    Start-Sleep -Seconds 3
}
