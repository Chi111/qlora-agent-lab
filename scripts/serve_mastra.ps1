param(
    [switch]$UseOllama
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$mastraRoot = Join-Path $projectRoot "mastra"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$node = (Get-Command node -ErrorAction SilentlyContinue).Source
Set-Location $projectRoot

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}
if (-not $UseOllama -and -not (Test-Path "artifacts\qlora-adapter\adapter_config.json")) {
    throw "QLoRA adapter not found. Run scripts\train.ps1 first."
}
if (-not (Test-Path (Join-Path $mastraRoot "node_modules"))) {
    throw "Mastra dependencies not found. Run scripts\setup_mastra.ps1 first."
}
$mastraEntry = Join-Path $mastraRoot "node_modules\mastra\dist\index.js"
$tsxCommand = Join-Path $mastraRoot "node_modules\.bin\tsx.cmd"
$viteEntry = Join-Path $mastraRoot "web\node_modules\vite\bin\vite.js"
if (-not $node) {
    throw "Node.js not found. Install Node.js 22.18 or newer."
}
if (-not (Test-Path $mastraEntry)) {
    throw "Mastra executable not found. Run scripts\setup_mastra.ps1 again."
}
if (-not (Test-Path $tsxCommand) -or -not (Test-Path $viteEntry)) {
    throw "RAG or web executables not found. Run scripts\setup_mastra.ps1 again."
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is required for the local Qdrant vector database."
}

$trainingConfigPath = "artifacts\qlora-adapter\training_config.json"
if (-not $UseOllama -and (Test-Path $trainingConfigPath)) {
    $trainingConfig = Get-Content $trainingConfigPath -Raw | ConvertFrom-Json
    $env:INFERENCE_BASE_MODEL_ID = $trainingConfig.base_model_id
    Write-Host "Using trained base model: $($env:INFERENCE_BASE_MODEL_ID)"
}
if ($UseOllama) {
    $env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434/v1"
    $env:LOCAL_LLM_MODEL = "qwen3.5:9b"
    $env:LOCAL_LLM_API_KEY = "ollama"
    $env:AGENT_MODEL_BASE_URL = "http://127.0.0.1:11434/v1"
    $env:AGENT_MODEL_NAME = "qwen3.5:9b"
    $env:AGENT_MODEL_API_KEY = "ollama"
}

function Get-ConfiguredValue {
    param(
        [string]$Key,
        [string]$EnvFile
    )
    $processValue = [Environment]::GetEnvironmentVariable($Key)
    if ($processValue) {
        return $processValue
    }
    if (-not (Test-Path $EnvFile)) {
        return $null
    }
    $match = Get-Content $EnvFile | Where-Object {
        $_ -match "^\s*$([regex]::Escape($Key))\s*="
    } | Select-Object -Last 1
    if (-not $match) {
        return $null
    }
    return ($match -split "=", 2)[1].Trim().Trim('"').Trim("'")
}

$agentRagKey = Get-ConfiguredValue -Key "AGENT_RAG_INTERNAL_API_KEY" -EnvFile (Join-Path $projectRoot ".env")
$mastraRagKey = Get-ConfiguredValue -Key "MASTRA_INTERNAL_SEARCH_KEY" -EnvFile (Join-Path $mastraRoot ".env")
if (-not $agentRagKey -or -not $mastraRagKey) {
    throw "RAG internal keys are missing. Update .env and mastra\.env from their .env.example files."
}
if ($agentRagKey -ne $mastraRagKey) {
    throw "AGENT_RAG_INTERNAL_API_KEY must exactly match MASTRA_INTERNAL_SEARCH_KEY."
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

function Stop-ProcessTree {
    param([System.Diagnostics.Process]$Process)
    if ($Process.HasExited) {
        return
    }
    if (Get-Command taskkill.exe -ErrorAction SilentlyContinue) {
        & taskkill.exe /PID $Process.Id /T /F | Out-Null
    }
    else {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

$processes = @()
try {
    Write-Host "Starting Qdrant on 127.0.0.1:6333..."
    & docker compose -f (Join-Path $projectRoot "compose.rag.yml") up -d qdrant
    if ($LASTEXITCODE -ne 0) {
        throw "Qdrant failed to start."
    }

    $processes += Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.mock_backend.app:app", "--host", "127.0.0.1", "--port", "8001", "--workers", "1" -PassThru
    if (-not $UseOllama) {
        $processes += Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.inference.app:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1" -PassThru
    }
    $processes += Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.retrieval.app:app", "--host", "127.0.0.1", "--port", "8003", "--workers", "1" -PassThru
    $processes += Start-Process -FilePath $python -ArgumentList "-m", "uvicorn", "qlora_lab.agent.app:app", "--host", "127.0.0.1", "--port", "8002", "--workers", "1" -PassThru

    Write-Host "Loading the language model, BGE-M3 and BGE Reranker. First startup may download models."
    Wait-ForService -Uri "http://127.0.0.1:6333/healthz" -TimeoutSeconds 60
    Wait-ForService -Uri "http://127.0.0.1:8001/health" -TimeoutSeconds 30
    if ($UseOllama) {
        Wait-ForService -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSeconds 60
    }
    else {
        Wait-ForService -Uri "http://127.0.0.1:8000/health" -TimeoutSeconds 240
    }
    Wait-ForService -Uri "http://127.0.0.1:8003/health" -TimeoutSeconds 600
    Wait-ForService -Uri "http://127.0.0.1:8002/health" -TimeoutSeconds 60

    Write-Host "Rebuilding the repair knowledge index and atomically switching its alias..."
    Set-Location $mastraRoot
    & $tsxCommand --env-file=.env src/mastra/rag/ingest.ts --rebuild
    if ($LASTEXITCODE -ne 0) {
        throw "Knowledge ingestion failed. Mastra was not started with a stale index."
    }

    $mastraProcess = Start-Process -FilePath $node -ArgumentList $mastraEntry, "dev" -WorkingDirectory $mastraRoot -PassThru
    $processes += $mastraProcess
    Wait-ForService -Uri "http://127.0.0.1:4111/api/agents" -TimeoutSeconds 120

    $webRoot = Join-Path $mastraRoot "web"
    $processes += Start-Process -FilePath $node -ArgumentList $viteEntry, "--host", "127.0.0.1", "--port", "5173", "--strictPort" -WorkingDirectory $webRoot -PassThru
    Wait-ForService -Uri "http://127.0.0.1:5173" -TimeoutSeconds 60

    $modelEndpoint = if ($UseOllama) { "Ollama=11434" } else { "inference=8000" }
    Write-Host "Services are ready: $modelEndpoint, mock=8001, LangChain=8002, retrieval=8003, Qdrant=6333"
    Write-Host "Customer service web: http://127.0.0.1:5173"
    Write-Host "Mastra Studio/API: http://127.0.0.1:4111"
    Write-Host "Press Ctrl+C to stop the services started by this script."
    Wait-Process -Id $mastraProcess.Id
    throw "Mastra exited unexpectedly. Check its process output and restart the stack."
}
finally {
    foreach ($process in $processes) {
        Stop-ProcessTree -Process $process
    }
    Set-Location $projectRoot
    & docker compose -f (Join-Path $projectRoot "compose.rag.yml") stop qdrant | Out-Null
}
