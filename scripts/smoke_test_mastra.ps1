param(
    [string]$Prompt = "冰箱不制冷应该怎么安全排查？",
    [string]$InternalSearchKey = "change-me-local-rag-key"
)

$ErrorActionPreference = "Stop"

Write-Host "Checking retrieval model and vector database..."
$retrieval = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8003/health"
if (-not $retrieval.models_loaded) {
    throw "Retrieval models are not loaded."
}
$qdrant = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:6333/healthz"

Write-Host "Checking mandatory rerank through the internal knowledge route..."
$searchBody = @{
    query = "冰箱不制冷"
    topK = 4
    knowledgeScope = "repair"
} | ConvertTo-Json
$search = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:4111/api/internal/knowledge/search" `
    -Headers @{ "x-internal-api-key" = $InternalSearchKey } `
    -ContentType "application/json; charset=utf-8" `
    -Body $searchBody
if (-not $search.ok -or -not $search.reranked -or $search.results.Count -lt 1) {
    throw "RAG smoke test failed or returned unreranked data."
}

Write-Host "Checking Mastra Agent registration..."
$agents = Invoke-RestMethod `
    -Method Get `
    -Uri "http://127.0.0.1:4111/api/agents"

if (-not $agents.'customer-service-agent') {
    throw "customer-service-agent is not registered."
}

Write-Host "Calling the local customer-service Agent..."
$body = @{
    messages = $Prompt
    maxSteps = 6
} | ConvertTo-Json -Depth 8

$response = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:4111/api/agents/customer-service-agent/generate" `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

$response | ConvertTo-Json -Depth 20

Write-Host "Checking React customer service web..."
$web = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:5173"
if ($web.StatusCode -ne 200) {
    throw "Customer service web is not available."
}

Write-Host "RAG, rerank, Agent and web smoke tests passed."
