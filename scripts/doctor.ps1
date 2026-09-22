& (Join-Path (Split-Path $PSScriptRoot -Parent) "vast-agent.ps1") doctor
exit $LASTEXITCODE
