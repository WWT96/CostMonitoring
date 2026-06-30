[CmdletBinding()]
param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$TaskName = "CostMonitoring Git Auto Sync",
    [int]$DebounceSeconds = 60,
    [int]$PollSeconds = 15
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$syncScript = Join-Path $RepoRoot "scripts\git_auto_sync.ps1"
if (-not (Test-Path -LiteralPath $syncScript)) {
    throw "Auto-sync script not found: $syncScript"
}

$powerShell = (Get-Command powershell.exe).Source
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$syncScript`" -RepoRoot `"$RepoRoot`" -DebounceSeconds $DebounceSeconds -PollSeconds $PollSeconds"

$action = New-ScheduledTaskAction -Execute $powerShell -Argument $arguments -WorkingDirectory $RepoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Auto commit and push CostMonitoring changes to GitHub." `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Installed and started task '$TaskName'. Logs: .git\auto-sync.log"
