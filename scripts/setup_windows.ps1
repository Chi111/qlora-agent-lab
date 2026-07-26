param(
    [string]$Python = "py -3.11",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "Creating Python 3.11 virtual environment..."
Invoke-Expression "$Python -m venv .venv"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip wheel
& $venvPython -m pip install torch --index-url $TorchIndexUrl
& $venvPython -m pip install -e ".[ml,agent,rag,dev]"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

$rootEnv = Get-Content ".env"
if (-not ($rootEnv | Where-Object { $_ -match "^\s*AGENT_RAG_INTERNAL_API_KEY\s*=" })) {
    throw ".env is missing AGENT_RAG_INTERNAL_API_KEY. Merge the new RAG values from .env.example and retry."
}

Write-Host ""
Write-Host "Running CUDA/QLoRA preflight..."
& $venvPython -m qlora_lab.training.preflight
Write-Host ""
Write-Host "Setup complete. Edit .env if needed, then run scripts\train.ps1."
