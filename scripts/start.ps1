$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

Write-Host 'Installing frontend dependencies...'
Push-Location -LiteralPath (Join-Path $root 'frontend')
try {
    & npm.cmd ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) {
        throw "npm ci failed with exit code $LASTEXITCODE"
    }

    Write-Host 'Building frontend...'
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) {
        throw "npm run build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host 'CircuitMind: http://127.0.0.1:8000/student'
& conda run -n llm python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

