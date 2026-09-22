param([Parameter(Mandatory=$true)][string]$Path)
& (Join-Path (Split-Path $PSScriptRoot -Parent) "vast-agent.ps1") migrate-v1 $Path
exit $LASTEXITCODE
