$ErrorActionPreference = "Stop"

Write-Host "Checking services..."
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8001/health" | ConvertTo-Json
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8002/health" | ConvertTo-Json
$ui = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8002/"
if ($ui.StatusCode -ne 200) {
    throw "Customer service UI is unavailable."
}

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

$chatBody = @{
    message = "显示器出现固定亮点怎么办？"
} | ConvertTo-Json

Write-Host "Calling the persistent chat/RAG flow..."
$chat = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8002/agent/chat" `
    -ContentType "application/json; charset=utf-8" `
    -Body $chatBody
$chat | ConvertTo-Json -Depth 12

Write-Host "Checking saved transcript..."
Invoke-RestMethod `
    -Method Get `
    -Uri "http://127.0.0.1:8002/agent/sessions/$($chat.session_id)" | ConvertTo-Json -Depth 12
