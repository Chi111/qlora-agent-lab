$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$mastraRoot = Join-Path $projectRoot "mastra"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
Set-Location $projectRoot

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}
if (-not (Test-Path "artifacts\qlora-adapter\adapter_config.json")) {
    throw "QLoRA adapter not found. Run scripts\train.ps1 first."
}
if (-not (Test-Path (Join-Path $mastraRoot "node_modules"))) {
    throw "Mastra dependencies not found. Run scripts\setup_mastra.ps1 first."
}
$mastraCommand = Join-Path $mastraRoot "node_modules\.bin\mastra.cmd"
if (-not (Test-Path $mastraCommand)) {
    throw "Mastra executable not found. Run scripts\setup_mastra.ps1 again."
}

$trainingConfigPath = "artifacts\qlora-adapter\training_config.json"
if (Test-Path $trainingConfigPath) {
    $trainingConfig = Get-Content $trainingConfigPath -Raw | ConvertFrom-Json
    $env:INFERENCE_BASE_MODEL_ID = $trainingConfig.base_model_id
    Write-Host "Using trained base model: $($env:INFERENCE_BASE_MODEL_ID)"
}

function Wait-ForService {
    param(
        [string]$Uri,
        [int]$TimeoutSeconds
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                return
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    } while ((Get-Date) -lt $deadline)
    throw "Service did not become ready in time: $Uri"
}

$processes = @()
try {
    $processes += Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.mock_backend.app:app", "--host", "127.0.0.1", "--port", "8001", "--workers", "1" -PassThru
    $processes += Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.inference.app:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1" -PassThru

    Write-Host "Loading the QLoRA model. This may take a minute..."
    Wait-ForService -Uri "http://127.0.0.1:8001/health" -TimeoutSeconds 30
    Wait-ForService -Uri "http://127.0.0.1:8000/health" -TimeoutSeconds 240

    Write-Host "Python services are ready: inference=8000, mock=8001"
    Write-Host "Starting Mastra Studio: http://127.0.0.1:4111"
    Write-Host "Press Ctrl+C to stop the services started by this script."
    Set-Location $mastraRoot
    & $mastraCommand dev
    if ($LASTEXITCODE -ne 0) {
        throw "Mastra exited with code $LASTEXITCODE."
    }
}
finally {
    foreach ($process in $processes) {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
        }
    }
}
