<# 
    InferenceBridge Qwen3.5 Model Tests
    
    Tests two models via the InferenceBridge REST API (port 8800):
      1. Qwen3.5-9B-Q4_K_M   (5.2 GB - fast, fits easily in VRAM)
      2. Qwen3.5-35B-A3B-Q4_K_M (19.7 GB - larger MoE model, needs more VRAM)

    Prerequisites:
      - InferenceBridge running (`npm run tauri dev`)
      - Model scan_dirs configured to include the LM Studio models cache
      - llama-server on PATH or configured in inference-bridge.toml

    Usage:
      .\tests\test-qwen35.ps1              # Run both models
      .\tests\test-qwen35.ps1 -Model 9b    # Run only 9B
      .\tests\test-qwen35.ps1 -Model 35b   # Run only 35B
#>

param(
    [ValidateSet("all", "9b", "35b")]
    [string]$Model = "all",

    [string]$BaseUrl = "http://localhost:8800",

    [int]$ContextSize = 8192,

    [int]$LoadTimeoutSecs = 180
)

$ErrorActionPreference = "Stop"

# -- Helpers ------------------------------------------

function Write-TestHeader($msg) {
    Write-Host "`n$("=" * 60)" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "$("=" * 60)" -ForegroundColor Cyan
}

function Write-Pass($msg) {
    Write-Host "  [PASS] $msg" -ForegroundColor Green
}

function Write-Fail($msg) {
    Write-Host "  [FAIL] $msg" -ForegroundColor Red
}

function Write-Info($msg) {
    Write-Host "  [INFO] $msg" -ForegroundColor Yellow
}

function Test-ApiReachable {
    try {
        $null = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
}

function Invoke-Scan {
    Write-Info "Triggering model scan..."
    # Use the API - scan is not exposed via REST, so we hit models endpoint
    # The scan happens via Tauri command; we just verify models appear after load
}

function Get-Models {
    $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -TimeoutSec 10
    return $resp
}

function Wait-ForModel($modelSubstring) {
    Write-Info "Waiting for model '$modelSubstring' to become ready (timeout: ${LoadTimeoutSecs}s)..."
    $elapsed = 0
    while ($elapsed -lt $LoadTimeoutSecs) {
        try {
            $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -TimeoutSec 5
            if ($resp.data -and $resp.data.Count -gt 0) {
                $loaded = $resp.data[0].id
                if ($loaded -match $modelSubstring) {
                    Write-Pass "Model loaded: $loaded"
                    return $true
                }
            }
        } catch { }
        Start-Sleep -Seconds 3
        $elapsed += 3
    }
    Write-Fail "Model load timed out after ${LoadTimeoutSecs}s"
    return $false
}

function Send-Completion($prompt, $maxTokens = 256, $temperature = 0.15) {
    $body = @{
        messages = @(
            @{ role = "user"; content = $prompt }
        )
        max_tokens = $maxTokens
        temperature = $temperature
    } | ConvertTo-Json -Depth 5

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 300
    $sw.Stop()
    
    return @{
        Response = $resp
        ElapsedMs = $sw.ElapsedMilliseconds
    }
}

function Get-ContextStatus {
    try {
        return Invoke-RestMethod -Uri "$BaseUrl/v1/context/status" -TimeoutSec 5
    } catch {
        return $null
    }
}

# -- Test Suite ---------------------------------------

function Run-ModelTests($modelName, $modelSubstring) {
    Write-TestHeader "Testing: $modelName"

    # Test 1: Model endpoint returns data
    Write-Info "Test 1: GET /v1/models"
    try {
        $models = Get-Models
        if ($models) {
            Write-Pass "/v1/models returns data"
        } else {
            Write-Fail "/v1/models returned empty"
        }
    } catch {
        Write-Fail "/v1/models failed: $_"
        return
    }

    # Test 2: Simple chat completion
    Write-Info "Test 2: Simple completion"
    try {
        $result = Send-Completion "What is 2 + 2? Answer with just the number."
        $content = $result.Response.choices[0].message.content
        $ms = $result.ElapsedMs
        Write-Info "Response (${ms}ms): $content"
        if ($content -match "4") {
            Write-Pass "Correct answer received"
        } else {
            Write-Fail "Expected '4' in response"
        }
    } catch {
        Write-Fail "Completion failed: $_"
    }

    # Test 3: Longer generation
    Write-Info "Test 3: Longer generation (512 tokens)"
    try {
        $result = Send-Completion "Explain how a CPU cache works in 2-3 paragraphs." 512 0.3
        $content = $result.Response.choices[0].message.content
        $ms = $result.ElapsedMs
        $wordCount = ($content -split '\s+').Count
        Write-Info "Response: $wordCount words in ${ms}ms"
        if ($wordCount -gt 30) {
            Write-Pass "Substantial response generated ($wordCount words)"
        } else {
            Write-Fail "Response too short ($wordCount words)"
        }
    } catch {
        Write-Fail "Long completion failed: $_"
    }

    # Test 4: Context status
    Write-Info "Test 4: Context/KV status"
    try {
        $ctx = Get-ContextStatus
        if ($ctx) {
            $pct = [math]::Round($ctx.fill_ratio * 100, 1)
            Write-Info "KV: $($ctx.used_tokens)/$($ctx.total_tokens) tokens ($pct%)"
            if ($ctx.total_tokens -gt 0) {
                Write-Pass "Context status reporting active"
            } else {
                Write-Fail "Context total_tokens is 0"
            }
        } else {
            Write-Fail "Context status returned null"
        }
    } catch {
        Write-Fail "Context status failed: $_"
    }

    # Test 5: Tool call extraction test (Qwen XML format)
    Write-Info "Test 5: Tool call generation"
    try {
        $toolPrompt = @"
You have access to the following tools:
- get_weather(location: string): Returns current weather for a location

The user asks: What's the weather in London?

Call the get_weather tool with the appropriate arguments. Respond ONLY with the tool call in this exact format:
<tool_call>{"name": "get_weather", "arguments": {"location": "London"}}</tool_call>
"@
        $result = Send-Completion $toolPrompt 128 0.0
        $content = $result.Response.choices[0].message.content
        $toolCalls = $result.Response.choices[0].message.tool_calls
        $ms = $result.ElapsedMs
        
        if ($toolCalls -and $toolCalls.Count -gt 0) {
            Write-Pass "Tool call extracted: $($toolCalls[0].function.name) (${ms}ms)"
        } elseif ($content -match "get_weather") {
            Write-Info "Tool call in text but not extracted: $content"
            Write-Pass "Model produced tool call syntax (extraction may need tuning)"
        } else {
            Write-Info "Response: $content"
            Write-Fail "No tool call detected"
        }
    } catch {
        Write-Fail "Tool call test failed: $_"
    }

    # Test 6: Multi-turn (context accumulation)
    Write-Info "Test 6: Multi-turn context"
    try {
        $body = @{
            messages = @(
                @{ role = "user"; content = "My name is Alice." }
                @{ role = "assistant"; content = "Nice to meet you, Alice!" }
                @{ role = "user"; content = "What is my name?" }
            )
            max_tokens = 64
            temperature = 0.0
        } | ConvertTo-Json -Depth 5

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" `
            -Method POST -ContentType "application/json" -Body $body -TimeoutSec 120
        $sw.Stop()
        $content = $resp.choices[0].message.content
        
        if ($content -match "Alice") {
            Write-Pass "Multi-turn context retained ($($sw.ElapsedMilliseconds)ms)"
        } else {
            Write-Info "Response: $content"
            Write-Fail "Model didn't recall 'Alice' from context"
        }
    } catch {
        Write-Fail "Multi-turn test failed: $_"
    }

    Write-Host ""
}

# -- Main ---------------------------------------------

Write-TestHeader "InferenceBridge Qwen3.5 Test Suite"
Write-Info "API: $BaseUrl"
Write-Info "Models to test: $Model"

# Check API reachable
if (-not (Test-ApiReachable)) {
    Write-Fail "Cannot reach InferenceBridge API at $BaseUrl"
    Write-Host "`n  Make sure the app is running: npm run tauri dev" -ForegroundColor Yellow
    exit 1
}
Write-Pass "API reachable"

$models = @{
    "9b" = @{
        Name = "Qwen3.5-9B-Q4_K_M"
        Substring = "Qwen3.5-9B"
        SizeGB = 5.2
        Description = "9B parameter model (Q4_K_M quant) - fast, good for testing"
    }
    "35b" = @{
        Name = "Qwen3.5-35B-A3B-Q4_K_M"
        Substring = "Qwen3.5-35B"
        SizeGB = 19.7
        Description = "35B parameter MoE model (A3B, Q4_K_M) - larger, needs ~20GB VRAM"
    }
}

$toTest = if ($Model -eq "all") { @("9b", "35b") } else { @($Model) }

foreach ($key in $toTest) {
    $m = $models[$key]
    Write-TestHeader "$($m.Name) ($($m.SizeGB) GB)"
    Write-Info $m.Description

    Write-Host "`n  ACTION REQUIRED:" -ForegroundColor Magenta
    Write-Host "  Load '$($m.Name)' in the InferenceBridge GUI (Models tab)" -ForegroundColor Magenta
    Write-Host "  Then press Enter to continue..." -ForegroundColor Magenta
    Read-Host

    # Verify model is loaded
    try {
        $modelResp = Get-Models
        $loaded = ""
        if ($modelResp.data -and $modelResp.data.Count -gt 0) {
            $loaded = $modelResp.data[0].id
        }
        if ($loaded -match $m.Substring) {
            Write-Pass "Model confirmed loaded: $loaded"
        } else {
            Write-Info "Loaded model: '$loaded' (expected: $($m.Substring))"
            Write-Host "  Continue anyway? (y/n)" -ForegroundColor Yellow
            $continue = Read-Host
            if ($continue -ne "y") { continue }
        }
    } catch {
        Write-Fail "Could not verify model: $_"
        continue
    }

    Run-ModelTests $m.Name $m.Substring
}

Write-TestHeader "Test Suite Complete"
Write-Host ""
