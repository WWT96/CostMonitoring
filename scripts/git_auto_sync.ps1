[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$Remote = "origin",
    [int]$DebounceSeconds = 60,
    [int]$PollSeconds = 15,
    [switch]$Once,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Git {
    param([string[]]$Arguments)

    $output = & git @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed ($exitCode): $($output -join [Environment]::NewLine)"
    }

    return @($output)
}

function Get-GitStatus {
    $output = & git status --porcelain 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "git status --porcelain failed ($exitCode): $($output -join [Environment]::NewLine)"
    }

    return @($output)
}

function Write-AutoSyncLog {
    param([string]$Message)

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $script:LogPath -Value "[$stamp] $Message" -Encoding UTF8
}

function Test-StagedDiffHasSecret {
    $diff = & git diff --cached --unified=0 -- . 2>&1
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "git diff --cached failed ($exitCode): $($diff -join [Environment]::NewLine)"
    }

    $patterns = @(
        'sk-[A-Za-z0-9_-]{20,}',
        'ghp_[A-Za-z0-9_]{20,}',
        'github_pat_[A-Za-z0-9_]{20,}',
        '(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*["'']?[^"'']{8,}'
    )

    foreach ($line in @($diff)) {
        if (-not $line.StartsWith("+")) {
            continue
        }
        if ($line.StartsWith("+++")) {
            continue
        }

        foreach ($pattern in $patterns) {
            if ($line -match $pattern) {
                return $true
            }
        }
    }

    return $false
}

function Clear-StagedChanges {
    try {
        Invoke-Git @("reset", "-q", "--", ".") | Out-Null
    }
    catch {
        Write-AutoSyncLog "Could not unstage auto-sync changes: $($_.Exception.Message)"
    }
}

function Invoke-AutoSync {
    if (Test-Path -LiteralPath $script:LockPath) {
        Write-AutoSyncLog "Skipped: lock file already exists."
        return
    }

    New-Item -ItemType File -Path $script:LockPath -Force | Out-Null
    try {
        Set-Location -LiteralPath $script:RepoRoot
        $status = @(Get-GitStatus)
        if ($status.Count -eq 0) {
            return
        }

        if ($DryRun) {
            Write-AutoSyncLog "Dry run: changes detected, no commit created."
            return
        }

        Invoke-Git @("add", "-A", "--", ".") | Out-Null

        $stagedFiles = & git diff --cached --name-only -- . 2>&1
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "git diff --cached --name-only failed ($exitCode): $($stagedFiles -join [Environment]::NewLine)"
        }

        if (@($stagedFiles).Count -eq 0) {
            Write-AutoSyncLog "Skipped: no staged changes after git add."
            return
        }

        if (Test-StagedDiffHasSecret) {
            Clear-StagedChanges
            Write-AutoSyncLog "Skipped: secret-like content found in staged diff."
            return
        }

        $branch = (Invoke-Git @("branch", "--show-current") | Select-Object -First 1).Trim()
        if ([string]::IsNullOrWhiteSpace($branch)) {
            Clear-StagedChanges
            Write-AutoSyncLog "Skipped: detached HEAD."
            return
        }

        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $oldSkip = $env:COST_MONITORING_SKIP_AUTO_PUSH
        $env:COST_MONITORING_SKIP_AUTO_PUSH = "1"
        try {
            Invoke-Git @("commit", "-m", "Auto-sync: $timestamp") | Out-Null
        }
        finally {
            if ($null -eq $oldSkip) {
                Remove-Item Env:\COST_MONITORING_SKIP_AUTO_PUSH -ErrorAction SilentlyContinue
            }
            else {
                $env:COST_MONITORING_SKIP_AUTO_PUSH = $oldSkip
            }
        }

        Invoke-Git @("push", "-u", $Remote, "HEAD:$branch") | Out-Null
        Write-AutoSyncLog "Committed and pushed $branch."
    }
    catch {
        Write-AutoSyncLog "ERROR: $($_.Exception.Message)"
    }
    finally {
        Remove-Item -LiteralPath $script:LockPath -Force -ErrorAction SilentlyContinue
    }
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$script:RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
Set-Location -LiteralPath $script:RepoRoot
$null = Invoke-Git @("rev-parse", "--is-inside-work-tree")
$gitDir = (Invoke-Git @("rev-parse", "--git-dir") | Select-Object -First 1).Trim()
if (-not [System.IO.Path]::IsPathRooted($gitDir)) {
    $gitDir = Join-Path $script:RepoRoot $gitDir
}

$script:LogPath = Join-Path $gitDir "auto-sync.log"
$script:LockPath = Join-Path $gitDir "auto-sync.lock"

Write-AutoSyncLog "Auto-sync started. repo=$script:RepoRoot remote=$Remote debounce=${DebounceSeconds}s poll=${PollSeconds}s"

if ($Once) {
    Invoke-AutoSync
    exit 0
}

while ($true) {
    Start-Sleep -Seconds $PollSeconds
    try {
        $status = @(Get-GitStatus)
        if ($status.Count -eq 0) {
            continue
        }

        Write-AutoSyncLog "Change detected; waiting ${DebounceSeconds}s before sync."
        Start-Sleep -Seconds $DebounceSeconds
        Invoke-AutoSync
    }
    catch {
        Write-AutoSyncLog "ERROR: $($_.Exception.Message)"
        Start-Sleep -Seconds 10
    }
}
