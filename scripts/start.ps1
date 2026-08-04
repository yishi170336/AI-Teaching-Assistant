$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

function Resolve-CircuitMindPython {
    if ($env:CIRCUITMIND_PYTHON) {
        return $env:CIRCUITMIND_PYTHON
    }

    if ($env:VIRTUAL_ENV) {
        $venvPython = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
        if (Test-Path -LiteralPath $venvPython) {
            return $venvPython
        }
    }

    if (
        $env:CONDA_PREFIX -and
        $env:CONDA_DEFAULT_ENV -and
        $env:CONDA_DEFAULT_ENV -ne 'base'
    ) {
        $activePython = Join-Path $env:CONDA_PREFIX 'python.exe'
        if (Test-Path -LiteralPath $activePython) {
            return $activePython
        }
    }

    if (Get-Command conda -ErrorAction SilentlyContinue) {
        $condaOutput = @(
            & conda run -n llm python -c 'import sys; print(sys.executable)' 2>$null
        )
        if ($LASTEXITCODE -eq 0) {
            foreach ($line in $condaOutput) {
                $candidate = "$line".Trim()
                if ($candidate -and (Test-Path -LiteralPath $candidate)) {
                    return $candidate
                }
            }
        }
    }

    throw 'Cannot find the Conda environment "llm". Run "conda activate llm" first or set CIRCUITMIND_PYTHON.'
}

$python = Resolve-CircuitMindPython
Write-Host "Using Python: $python"
& $python -c 'import uvicorn' 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "uvicorn is not installed for $python. Install the backend dependencies in the selected virtual environment."
}

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
& $python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

