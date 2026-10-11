# run_local.ps1 — start the web application on Windows for local testing.
#
# Usage, from Artefact\webapp in your conda environment:
#     .\run_local.ps1            development server, auto-reload on template edits
#     .\run_local.ps1 -Test      run the web test suite first, then start
#
# If PowerShell refuses to run scripts:
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

param([switch]$Test)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:EGG_ARTEFACT_ROOT = (Resolve-Path (Join-Path $here "..")).Path
$env:FLASK_ENV = "development"
$env:SECRET_KEY = "local-development-only"
$env:TORCH_NUM_THREADS = "4"

Write-Host "Artefact root : $env:EGG_ARTEFACT_ROOT"

$ckpts = Get-ChildItem (Join-Path $env:EGG_ARTEFACT_ROOT "models\final") -Filter "final_*.pt" -ErrorAction SilentlyContinue
Write-Host "Checkpoints   : $($ckpts.Count) found"
if ($ckpts.Count -eq 0) { Write-Warning "No checkpoints in Artefact\models\final. Run notebook 07 first." }

$rules = Join-Path $env:EGG_ARTEFACT_ROOT "data\processed\root_cause_rules_recovered.csv"
if (-not (Test-Path $rules)) { Write-Warning "Rule base not found at $rules. Run notebook 01 first." }

if ($Test) {
    Write-Host "`nRunning web tests..." -ForegroundColor Cyan
    python -m pytest (Join-Path $here "tests") -q
    if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed; not starting the server." }
}

Write-Host "`nStarting on http://127.0.0.1:8000  (Ctrl+C to stop)`n" -ForegroundColor Green
python (Join-Path $here "wsgi.py")
