<#
    InferenceBridge GUI + API smoke test

    Goal:
      Verify that the desktop GUI can run while exposing an LM Studio style
      OpenAI-compatible API on port 8800 for external clients.

    What this checks:
      1. GUI-launched API responds on /v1/health
      2. /v1/models is reachable
      3. If a model is selected, /v1/chat/completions works
      4. Model activation through the API is reflected by the shared app state

    Typical usage:
      .\tests\test-gui-api-smoke.ps1
      .\tests\test-gui-api-smoke.ps1 -LaunchGui
      .\tests\test-gui-api-smoke.ps1 -Model "Qwen3.5-9B"
      .\tests\test-gui-api-smoke.ps1 -Model "Qwen3.5-9B" -Prompt "Reply with exactly: InferenceBridge OK"

    Notes:
      - This script assumes the public API lives at http://127.0.0.1:8800 by default.
      - If no model is loaded and no -Model is provided, the script only performs API
        reachability checks.
      - There is no public scan endpoint yet, so if /v1/models is empty you may need to
        scan from the GUI first.
#>

param(
    [string]$BaseUrl = "http://127.0.0.1:8800",
    [string]$Model,
    [string]$Prompt = "Reply with exactly: InferenceBridge OK",
    [int]$WaitForApiSecs = 20,
    [int]$CompletionTimeoutSecs = 300,
    [switch]$LaunchGui,
    [string]$ExePath = ".\src-tauri\target\release\inference-bridge.exe",
    [switch]$SkipChat
)

$ErrorActionPreference = "Stop"
$script:Failures = 0
$script:GuiProcess = $null

function Write-Section($msg) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host ("  " + $msg) -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
}

function Write-Pass($msg) {
    Write-Host "[PASS] $msg" -ForegroundColor Green
}

function Write-Info($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Yellow
}

function Write-WarnMsg($msg) {
    Write-Host "[WARN] $msg" -ForegroundColor DarkYellow
}

function Write-Fail($msg) {
    $script:Failures += 1
    Write-Host "[FAIL] $msg" -ForegroundColor Red
}

function Invoke-ApiJson {
    param(
        [Parameter(Mandatory = $true)][string]$Method,
        [Parameter(Mandatory = $true)][string]$Path,
        [object]$Body,
        [int]$TimeoutSec = 15
    )

    $params = @{
        Uri         = "$BaseUrl$Path"
        Method      = $Method
        TimeoutSec  = $TimeoutSec
        ErrorAction = "Stop"
    }

    if ($null -ne $Body) {
        $params.ContentType = "application/json"
        $params.Body = $Body | ConvertTo-Json -Depth 10
    }

    Invoke-RestMethod @params
}

function Wait-ForApi {
    $deadline = (Get-Date).AddSeconds($WaitForApiSecs)
    while ((Get-Date) -lt $deadline) {
        try {
            $health = Invoke-ApiJson -Method GET -Path "/v1/health" -TimeoutSec 3
            return $health
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "Timed out waiting for API at $BaseUrl/v1/health"
}

function Get-ModelsResponse {
    Invoke-ApiJson -Method GET -Path "/v1/models" -TimeoutSec 10
}

function Get-ModelStats {
    Invoke-ApiJson -Method GET -Path "/v1/models/stats" -TimeoutSec 10
}

function Get-ActiveModelId {
    $models = Get-ModelsResponse
    $active = @($models.data | Where-Object { $_.active })
    if ($active.Count -gt 0) {
        return $active[0].id
    }
    return $null
}

function Resolve-ModelId {
    param([string]$RequestedModel)

    $models = Get-ModelsResponse
    $items = @($models.data)
    if ($items.Count -eq 0) {
        return $null
    }

    if ([string]::IsNullOrWhiteSpace($RequestedModel)) {
        $active = $items | Where-Object { $_.active } | Select-Object -First 1
        if ($null -ne $active) {
            return $active.id
        }
        return $items[0].id
    }

    $match = $items | Where-Object {
        $_.id -like "*$RequestedModel*"
    } | Select-Object -First 1

    return $match.id
}

function Invoke-ChatCompletion {
    param(
        [string]$ModelId,
        [string]$PromptText
    )

    $body = @{
        model       = $ModelId
        messages    = @(
            @{
                role    = "user"
                content = $PromptText
            }
        )
        max_tokens  = 96
        temperature = 0.0
        stream      = $false
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-ApiJson -Method POST -Path "/v1/chat/completions" -Body $body -TimeoutSec $CompletionTimeoutSecs
    $sw.Stop()

    @{
        Response  = $resp
        ElapsedMs = $sw.ElapsedMilliseconds
    }
}

function Start-GuiIfRequested {
    if (-not $LaunchGui) {
        return
    }

    if (-not (Test-Path -LiteralPath $ExePath)) {
        throw "GUI executable not found: $ExePath"
    }

    Write-Info "Launching GUI from $ExePath"
    $script:GuiProcess = Start-Process -FilePath $ExePath -PassThru
    Write-Info "Started GUI PID $($script:GuiProcess.Id)"
}

function Stop-GuiIfStarted {
    if ($null -eq $script:GuiProcess) {
        return
    }

    try {
        if (-not $script:GuiProcess.HasExited) {
            Write-Info "Stopping GUI process $($script:GuiProcess.Id)"
            Stop-Process -Id $script:GuiProcess.Id -Force
        }
    } catch {
        Write-WarnMsg "Could not stop launched GUI process: $_"
    }
}

try {
    Start-GuiIfRequested

    Write-Section "API Reachability"
    $health = Wait-ForApi
    Write-Pass "Public API reachable at $BaseUrl/v1"
    Write-Info "Health status: $($health.status)"
    if ($health.model) {
        Write-Info "Health model: $($health.model)"
    }

    Write-Section "Model Registry"
    $models = Get-ModelsResponse
    $items = @($models.data)
    Write-Pass "/v1/models responded successfully"
    Write-Info "Models reported: $($items.Count)"

    $activeBefore = Get-ActiveModelId
    if ($activeBefore) {
        Write-Info "Active model before test: $activeBefore"
    } else {
        Write-WarnMsg "No active model reported"
    }

    if ($items.Count -eq 0) {
        Write-WarnMsg "No models are registered yet. Scan model directories in the GUI first if you want to test completions."
    } else {
        $preview = ($items | Select-Object -First 8 | ForEach-Object { $_.id }) -join ", "
        Write-Info "Available models: $preview"
    }

    if ($SkipChat) {
        Write-Info "Skipping chat test by request"
    } elseif ($items.Count -eq 0) {
        Write-Info "Skipping chat test because no models are registered"
    } else {
        Write-Section "Chat Completion"

        $modelId = Resolve-ModelId -RequestedModel $Model
        if ([string]::IsNullOrWhiteSpace($modelId)) {
            if ($Model) {
                Write-Fail "Requested model '$Model' was not found in /v1/models"
            } else {
                Write-Fail "Could not resolve any model for completion"
            }
        } else {
            Write-Info "Using model: $modelId"
            $result = Invoke-ChatCompletion -ModelId $modelId -PromptText $Prompt
            $response = $result.Response
            $content = $response.choices[0].message.content
            Write-Pass "/v1/chat/completions returned a response in $($result.ElapsedMs) ms"
            Write-Info "Completion text: $content"

            $activeAfter = Get-ActiveModelId
            if ($activeAfter -eq $modelId) {
                Write-Pass "Shared app state reflects active model via /v1/models: $activeAfter"
            } else {
                Write-Fail "Expected active model '$modelId' after completion, but API reports '$activeAfter'"
            }

            $stats = Get-ModelStats
            if ($stats.progress) {
                Write-Info ("Model state: {0} ({1})" -f $stats.progress.stage, $stats.progress.message)
            } elseif ($stats.state) {
                Write-Info ("Model state: {0}" -f ($stats.state | ConvertTo-Json -Compress))
            }
        }
    }

    Write-Section "Summary"
    if ($script:Failures -eq 0) {
        Write-Pass "GUI + embedded API smoke test passed"
        exit 0
    } else {
        Write-Fail "$script:Failures check(s) failed"
        exit 1
    }
}
finally {
    Stop-GuiIfStarted
}
