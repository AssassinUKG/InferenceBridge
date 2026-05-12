param(
    [ValidateSet("Release", "Dev", "Check")]
    [string]$Mode = "Release",

    [switch]$SkipInstall,

    [switch]$CleanInstall
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
        Step "Building desktop release bundle"
        Run "npm.cmd" @("run", "tauri", "build")

        Write-Host ""
        Write-Host "InferenceBridge build complete." -ForegroundColor Green
        Write-Host "Installers and bundles are in: src-tauri\target\release\bundle" -ForegroundColor Green
    }
}
