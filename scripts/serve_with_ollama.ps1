$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Set-Location $projectRoot

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}

$env:AGENT_MODEL_BASE_URL = "http://127.0.0.1:11434/v1"
$env:AGENT_MODEL_NAME = "qwen3.5:9b"
$env:AGENT_MODEL_API_KEY = "ollama"

$mock = Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.mock_backend.app:app", "--host", "127.0.0.1", "--port", "8001", "--workers", "1" -PassThru
Start-Sleep -Seconds 2
& $python -m uvicorn qlora_lab.agent.app:app --host 127.0.0.1 --port 8002 --workers 1

Write-Host "Mock backend process ID: $($mock.Id)"

