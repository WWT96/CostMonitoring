[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$TaskName = "CostMonitoring Git Auto Sync",
    [int]$DebounceSeconds = 60,
    [int]$PollSeconds = 15
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$syncScript = Join-Path $RepoRoot "scripts\git_auto_sync.ps1"
if (-not (Test-Path -LiteralPath $syncScript)) {
    throw "Auto-sync script not found: $syncScript"
}

$powerShell = (Get-Command powershell.exe).Source
$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$syncScript`" -RepoRoot `"$RepoRoot`" -DebounceSeconds $DebounceSeconds -PollSeconds $PollSeconds"

function Install-StartupShortcut {
    $startupDir = [Environment]::GetFolderPath("Startup")
    $shortcutPath = Join-Path $startupDir "$TaskName.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $powerShell
    $shortcut.Arguments = "-WindowStyle Hidden $arguments"
    $shortcut.WorkingDirectory = $RepoRoot
    $shortcut.WindowStyle = 7
    $shortcut.Description = "Auto commit and push CostMonitoring changes to GitHub."
    $shortcut.Save()

    Start-Process -FilePath $powerShell -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden
    Write-Host "Installed Startup shortcut '$shortcutPath' and started auto-sync. Logs: .git\auto-sync.log"
}

try {
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
}
catch {
    Write-Warning "Scheduled task install failed: $($_.Exception.Message)"
    Install-StartupShortcut
}
