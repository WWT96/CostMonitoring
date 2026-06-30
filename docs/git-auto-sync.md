# Git Auto Sync

This repository has two local automation layers for GitHub sync.

## Post-Commit Auto Push

`.githooks/post-commit` pushes the current branch to `origin` after every successful local commit. This works with GitHub Desktop because GitHub Desktop uses Git commits under the hood.

Enable it in this checkout:

```powershell
git config core.hooksPath .githooks
```

Temporarily skip one auto-push:

```powershell
$env:COST_MONITORING_SKIP_AUTO_PUSH = "1"
git commit -m "Your message"
Remove-Item Env:\COST_MONITORING_SKIP_AUTO_PUSH
```

Hook logs are written to `.git/post-commit-auto-push.log`.

## Background Auto Commit And Push

For hands-off syncing, install the Windows scheduled task:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_git_auto_sync_task.ps1
```

If Windows denies scheduled task registration, the installer falls back to a user Startup shortcut:

```text
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\CostMonitoring Git Auto Sync.lnk
```

That shortcut starts the same background sync script when the current Windows user logs in.

The task starts at Windows logon and watches this repo by polling `git status`. When tracked or untracked non-ignored files change, it waits for a debounce window, then runs:

```text
git add -A
git commit -m "Auto-sync: yyyy-MM-dd HH:mm:ss"
git push -u origin HEAD:<current-branch>
```

The script skips commits if the staged diff contains obvious secret-like tokens, GitHub tokens, or OpenAI-style keys. Keep private files in `.gitignore`; automatic sync is only as safe as the ignore rules.

Remove the task:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\uninstall_git_auto_sync_task.ps1
```

The uninstall script removes either setup: the scheduled task if present, and the Startup shortcut if present.

Background sync logs are written to `.git/auto-sync.log`.
