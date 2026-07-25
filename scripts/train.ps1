param(
    [string]$Config = "configs/train_8gb.yaml"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Set-Location $projectRoot

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}

Write-Host "Stop Ollama and other GPU-heavy applications before training."
& $python -m qlora_lab.training.preflight
& $python -m qlora_lab.training.train --config $Config

