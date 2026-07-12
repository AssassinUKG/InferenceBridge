param(
    [ValidateSet("Release", "Dev", "Check")]
    [string]$Mode = "Release",

    [switch]$SkipInstall,

    [switch]$CleanInstall,

    [switch]$KeepRunning,

    [switch]$Run
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

function Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Run($Command, [string[]]$Arguments) {
    Write-Host "    $Command $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

function Stop-RunningReleaseBinary {
    if ($KeepRunning) {
        return $false
    }

    $releaseExe = Join-Path $PSScriptRoot "src-tauri\target\release\inference-bridge.exe"
    $releaseExeLower = $releaseExe.ToLowerInvariant()
    $locked = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -ieq "inference-bridge.exe" -and
            $_.CommandLine -and
            $_.CommandLine.ToLowerInvariant().Contains($releaseExeLower)
        }

    $wasRunning = $false
    if ($locked) {
        $wasRunning = $true
        Step "Stopping running release binary before build"
        foreach ($proc in $locked) {
            Write-Host "    stopping inference-bridge.exe pid=$($proc.ProcessId)" -ForegroundColor DarkGray
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 500
    }

    # The managed llama-server is usually a child/runtime companion of the bridge.
    # Stop it too so a fresh post-build launch does not keep stale model/runtime args.
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    return $wasRunning
}

if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "npm.cmd was not found on PATH. Install Node.js 18+ before building InferenceBridge."
}

if (-not (Get-Command cargo.exe -ErrorAction SilentlyContinue)) {
    throw "cargo.exe was not found on PATH. Install Rust 1.75+ before building InferenceBridge."
}

if (-not $SkipInstall) {
    Step "Installing Node dependencies"
    if ($CleanInstall -and (Test-Path -LiteralPath "package-lock.json")) {
        Run "npm.cmd" @("ci")
    } else {
        Run "npm.cmd" @("install")
    }
}

switch ($Mode) {
    "Dev" {
        Step "Starting Tauri desktop dev app"
        Run "npm.cmd" @("run", "tauri", "dev")
    }

    "Check" {
        Step "Checking frontend"
        Run "npm.cmd" @("run", "build")

        Step "Checking Rust backend"
        Push-Location -LiteralPath "src-tauri"
        try {
            Run "cargo.exe" @("check")
        } finally {
            Pop-Location
        }

        Write-Host ""
        Write-Host "Checks passed." -ForegroundColor Green
    }

    "Release" {
        $wasRunning = Stop-RunningReleaseBinary

        Step "Building desktop release bundle"
        Run "npm.cmd" @("run", "tauri", "build")

        Write-Host ""
        Write-Host "InferenceBridge build complete." -ForegroundColor Green
        Write-Host "Installers and bundles are in: src-tauri\target\release\bundle" -ForegroundColor Green

        if ($Run -or $wasRunning) {
            $releaseExe = Join-Path $PSScriptRoot "src-tauri\target\release\inference-bridge.exe"
            Step "Starting rebuilt release binary"
            Start-Process -FilePath $releaseExe -WorkingDirectory $PSScriptRoot
        }
    }
}
