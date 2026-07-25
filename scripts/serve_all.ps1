$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Set-Location $projectRoot

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}
if (-not (Test-Path "artifacts\qlora-adapter\adapter_config.json")) {
    throw "QLoRA adapter not found. Run scripts\train.ps1 first."
}

$processes = @()
$processes += Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.mock_backend.app:app", "--host", "127.0.0.1", "--port", "8001", "--workers", "1" -PassThru
$processes += Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.inference.app:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1" -PassThru
Start-Sleep -Seconds 12
$processes += Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.agent.app:app", "--host", "127.0.0.1", "--port", "8002", "--workers", "1" -PassThru

Write-Host "Services started: inference=8000, mock=8001, agent=8002"
Write-Host "Process IDs: $($processes.Id -join ', ')"
Write-Host "Run scripts\smoke_test.ps1 in another PowerShell window."

