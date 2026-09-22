param([Parameter(Position=0)][string]$Command, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
if ($Command -eq "install") { & "$root\scripts\install.ps1"; exit $LASTEXITCODE }
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { Write-Error "Virtual environment missing. Run setup.cmd first."; exit 2 }
& $python -m vast_agent $Command @Rest
exit $LASTEXITCODE
