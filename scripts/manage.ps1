param([Parameter(Mandatory=$true)][string]$Command, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
& (Join-Path (Split-Path $PSScriptRoot -Parent) "vast-agent.ps1") $Command @Rest
exit $LASTEXITCODE
