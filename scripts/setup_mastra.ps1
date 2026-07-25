param(
    [string]$PnpmVersion = "11.0.6"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$mastraRoot = Join-Path $projectRoot "mastra"
Set-Location $mastraRoot

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js not found. Install Node.js 22.18 or newer first."
}

$nodeVersion = [version]((& node -p "process.versions.node").Trim())
if ($nodeVersion -lt [version]"22.18.0") {
    throw "Mastra requires Node.js 22.18 or newer. Current version: $nodeVersion"
}

function Invoke-ProjectPnpm {
    param([string[]]$PnpmArgs)
    if (-not (Get-Command npx -ErrorAction SilentlyContinue)) {
        throw "npx not found. Reinstall the official Node.js distribution and retry."
    }
    & npx --yes "pnpm@$PnpmVersion" @PnpmArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pnpm command failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

Write-Host "Installing pinned Mastra dependencies..."
Invoke-ProjectPnpm -PnpmArgs @("install", "--frozen-lockfile")
Write-Host "Building the approved esbuild binary..."
Invoke-ProjectPnpm -PnpmArgs @("rebuild", "esbuild")
Write-Host "Running TypeScript tests and build..."
Invoke-ProjectPnpm -PnpmArgs @("run", "check")
Write-Host ""
Write-Host "Mastra setup complete."
