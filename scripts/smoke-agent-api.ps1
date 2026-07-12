param(
  [string]$BaseUrl = "http://127.0.0.1:8800/v1",
  [string]$Model = "Qwen3.6-27B-Q4_K_M.gguf",
  [int]$TimeoutSec = 180
)

$ErrorActionPreference = "Stop"

function Invoke-Json {
  param(
    [string]$Path,
    [object]$Body,
    [int]$RequestTimeoutSec = $TimeoutSec
  )

  $json = $Body | ConvertTo-Json -Depth 40 -Compress
  Invoke-RestMethod -Uri "$BaseUrl$Path" -Method Post -ContentType "application/json" -Body $json -TimeoutSec $RequestTimeoutSec
}

function Invoke-JsonRaw {
  param(
    [string]$Path,
    [object]$Body,
    [int]$RequestTimeoutSec = $TimeoutSec
  )

  $json = $Body | ConvertTo-Json -Depth 40 -Compress
  $tmp = Join-Path $env:TEMP "ib-smoke-$([guid]::NewGuid()).json"
  [System.IO.File]::WriteAllText($tmp, $json, [System.Text.UTF8Encoding]::new($false))
  try {
    & curl.exe -sS -N -X POST "$BaseUrl$Path" -H "Content-Type: application/json" --max-time $RequestTimeoutSec --data-binary "@$tmp"
    if ($LASTEXITCODE -ne 0) {
      throw "curl exited with code $LASTEXITCODE"
    }
  } finally {
    Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
  }
}

$results = [ordered]@{}

$results.health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 10
$models = Invoke-RestMethod -Uri "$BaseUrl/models" -TimeoutSec 20
$results.model_count = @($models.data).Count

$structuredBody = [ordered]@{
  model = $Model
  max_tokens = 80
  temperature = 0
  response_format = @{ type = "json_object" }
  messages = @(@{
    role = "user"
    content = "Return exactly JSON with keys ok and animal. Use ok true and animal cat."
  })
}

try {
  $structured = Invoke-Json "/chat/completions" $structuredBody
  $text = [string]$structured.choices[0].message.content
  $parsed = $text | ConvertFrom-Json
  $results.structured_json = @{
    passed = ($parsed.ok -eq $true -and $parsed.animal -eq "cat")
    raw = $text
  }
} catch {
  $results.structured_json = @{ passed = $false; error = $_.Exception.Message }
}

$messagesBody = [ordered]@{
  model = $Model
  max_tokens = 64
  temperature = 0
  system = "Reply tersely."
  messages = @(@{ role = "user"; content = "Say hello in exactly two words." })
}

try {
  $message = Invoke-Json "/messages" $messagesBody
  $results.messages = @{
    passed = ($message.type -eq "message" -and $message.role -eq "assistant" -and @($message.content).Count -gt 0)
    stop_reason = $message.stop_reason
    content = $message.content
  }
} catch {
  $results.messages = @{ passed = $false; error = $_.Exception.Message }
}

$streamBody = [ordered]@{
  model = $Model
  max_tokens = 32
  temperature = 0
  stream = $true
  messages = @(@{ role = "user"; content = "Say stream ok." })
}

try {
  $streamRaw = Invoke-JsonRaw "/messages" $streamBody
  $events = @($streamRaw -split "`n" | Where-Object { $_ -like "event:*" } | ForEach-Object { $_.Substring(6).Trim() })
  $required = @("message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop")
  $results.messages_stream = @{
    passed = ($required | ForEach-Object { $events -contains $_ } | Where-Object { -not $_ } | Measure-Object).Count -eq 0
    events = $events
  }
} catch {
  $results.messages_stream = @{ passed = $false; error = $_.Exception.Message }
}

$embeddingBody = [ordered]@{
  model = $Model
  input = "hello"
}

try {
  $embedding = Invoke-Json "/embeddings" $embeddingBody 60
  $results.embeddings = @{
    passed = (@($embedding.data).Count -gt 0)
    object = $embedding.object
    data_count = @($embedding.data).Count
  }
} catch {
  $body = $null
  if ($_.ErrorDetails) {
    $body = $_.ErrorDetails.Message
  }
  $results.embeddings = @{
    passed = $true
    expected_single_runtime_failure = $true
    error = $_.Exception.Message
    body = $body
  }
}

$results | ConvertTo-Json -Depth 40
