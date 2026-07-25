$ErrorActionPreference = "Stop"

Write-Host "Checking services..."
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8001/health" | ConvertTo-Json
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8002/health" | ConvertTo-Json

$body = @{
    messages = @(
        @{ role = "user"; content = "帮我查询订单 A101。如果延迟了，先询问我是否创建工单。" }
    )
    debug = $true
} | ConvertTo-Json -Depth 8

Write-Host "Calling the agent..."
Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8002/agent/invoke" `
    -ContentType "application/json; charset=utf-8" `
    -Body $body | ConvertTo-Json -Depth 12

