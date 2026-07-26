$ErrorActionPreference = "Stop"

Write-Host "Starting the complete RAG stack with Ollama qwen3.5:9b as the language model."
& (Join-Path $PSScriptRoot "serve_mastra.ps1") -UseOllama
