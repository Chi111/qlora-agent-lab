param(
    [string]$Prompt = "你好，我的订单遇到问题了。"
)

$ErrorActionPreference = "Stop"

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
