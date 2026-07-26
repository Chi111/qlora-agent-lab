$ErrorActionPreference = "Stop"

Write-Host "serve_all.ps1 now starts the complete QLoRA + RAG + Mastra stack."
& (Join-Path $PSScriptRoot "serve_mastra.ps1")
