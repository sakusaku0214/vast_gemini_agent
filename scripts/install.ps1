$ErrorActionPreference = "Stop"
if (-not $IsWindows -and $PSVersionTable.PSEdition -eq "Core") { Write-Error "Vast Gemini Agent setup requires Windows 11."; exit 2 }
$missing = @()
foreach ($tool in @("git.exe", "ssh.exe", "ssh-keyscan.exe", "ssh-keygen.exe")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { $missing += $tool }
}
$py = Get-Command py.exe -ErrorAction SilentlyContinue
if (-not $py) { $missing += "Python launcher (py.exe) with Python 3.12" }
elseif (& py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null; $LASTEXITCODE -ne 0) { $missing += "Python 3.12" }
if ($missing.Count) {
    Write-Host "Setup cannot continue. Missing prerequisites:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host " - $_" }
    if (Get-Command winget.exe -ErrorAction SilentlyContinue) {
        Write-Host "winget is available. Install prerequisites explicitly, e.g. winget install Python.Python.3.12"
        Write-Host "OpenSSH Client can be enabled in Windows Optional Features."
    }
    exit 2
}
$root = Split-Path $PSScriptRoot -Parent
$venv = Join-Path $root ".venv"
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) { & py -3.12 -m venv $venv }
& (Join-Path $venv "Scripts\python.exe") -m pip install -e "${root}[dev]"
if ($LASTEXITCODE -ne 0) { Write-Error "Package installation failed."; exit $LASTEXITCODE }
& (Join-Path $venv "Scripts\python.exe") -m vast_agent install
if ($LASTEXITCODE -ne 0) { Write-Error "Runtime initialization failed."; exit $LASTEXITCODE }
Write-Host "Installation complete. Next: .\vast-agent.ps1 doctor" -ForegroundColor Green
