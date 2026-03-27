param(
    [string]$Url = "http://127.0.0.1:8000"
)

$model = "qwen3.5-35b"
$tokens = 200

Write-Host "[*] Testing: $Url"

# Build body safely (no broken quotes)
$bodyObj = @{
    model = $model
    messages = @(
        @{
            role = "user"
            content = "Explain privilege escalation in one paragraph."
        }
    )
    max_tokens = $tokens
    stream = $false
}

$body = $bodyObj | ConvertTo-Json -Depth 10

try {
    $start = Get-Date

    $response = Invoke-RestMethod `
        -Uri "$Url/v1/chat/completions" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body

    $end = Get-Date

    $time = ($end - $start).TotalSeconds

    # Safe token extraction
    $tokensOut = 0
    if ($response.usage -and $response.usage.completion_tokens) {
        $tokensOut = $response.usage.completion_tokens
    }

    if ($time -gt 0) {
        $tps = [math]::Round($tokensOut / $time, 2)
    } else {
        $tps = 0
    }

    Write-Host "-----------------------------"
    Write-Host "Time: $time sec"
    Write-Host "Tokens: $tokensOut"
    Write-Host "Tokens/sec: $tps"
    Write-Host "-----------------------------"

} catch {
    Write-Host "[!] Request failed:"
    Write-Host $_
}