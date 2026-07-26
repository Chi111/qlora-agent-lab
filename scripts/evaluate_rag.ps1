param(
    [string]$ApiKey = $env:AGENT_RAG_INTERNAL_API_KEY,
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts\setup_windows.ps1 first."
}
if (-not $ApiKey) {
    throw "Pass -ApiKey or set AGENT_RAG_INTERNAL_API_KEY."
}

$arguments = @((Join-Path $PSScriptRoot "evaluate_rag.py"), "--api-key", $ApiKey)
if ($Strict) {
    $arguments += "--strict"
}
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Live RAG evaluation did not meet the strict thresholds."
}
