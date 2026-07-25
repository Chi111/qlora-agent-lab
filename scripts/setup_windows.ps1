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
& $venvPython -m pip install -e ".[ml,agent,dev]"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
}

Write-Host ""
Write-Host "Running CUDA/QLoRA preflight..."
& $venvPython -m qlora_lab.training.preflight
Write-Host ""
Write-Host "Setup complete. Edit .env if needed, then run scripts\train.ps1."

