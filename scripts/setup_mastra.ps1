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

$requiredEnvKeys = @(
    "RAG_MODEL_SERVICE_URL",
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "MASTRA_INTERNAL_SEARCH_KEY",
    "RAG_REQUEST_TIMEOUT_MS",
    "RAG_MIN_RERANK_SCORE"
)
$envContent = Get-Content ".env"
$missingEnvKeys = $requiredEnvKeys | Where-Object {
    $key = $_
    -not ($envContent | Where-Object { $_ -match "^\s*$([regex]::Escape($key))\s*=" })
}
if ($missingEnvKeys.Count -gt 0) {
    throw "mastra\.env is missing: $($missingEnvKeys -join ', '). Merge the new values from mastra\.env.example and retry."
}

Write-Host "Installing pinned Mastra dependencies..."
Invoke-ProjectPnpm -PnpmArgs @("install", "--frozen-lockfile")
Write-Host "Building the approved esbuild binary..."
Invoke-ProjectPnpm -PnpmArgs @("rebuild", "esbuild")
Write-Host "Running TypeScript tests and build..."
Invoke-ProjectPnpm -PnpmArgs @("run", "check")
Write-Host ""
Write-Host "Mastra setup complete."
