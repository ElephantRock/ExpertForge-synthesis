# E0 Windows Qualification Bootstrap — Local Assistant Handoff

GitHub is authoritative. Treat this checkout as a disposable working copy.

- Repository: `ElephantRock/ExpertForge-synthesis`
- Working branch: `e0/qualification-bootstrap`
- Coordination issue: `#1 E0 qualification bootstrap on Windows`
- Scope: qualification/bootstrap only; no terminal E0 evidence execution.

## Start here

From PowerShell in the local checkout:

```powershell
git fetch origin
git checkout e0/qualification-bootstrap
git pull --ff-only origin e0/qualification-bootstrap
git status
git remote -v
git log -1 --oneline
```

Then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\e0\bootstrap_windows.ps1
```

Commit and push the generated environment snapshot before expensive model qualification begins.

## Operating rules

1. Read GitHub issue #1 before each phase. If local notes disagree with GitHub, stop and reconcile on GitHub.
2. Never edit frozen scientific thresholds, seeds, arm definitions, or statistical rules to fit local results.
3. Never commit credentials or downloaded model weights.
4. Commit each qualification phase separately and push frequently.
5. Q1 must reproduce locally before Q2/Q3 begin.
6. Stop on contract/integrity contradictions; report them in issue #1 rather than improvising a scientific change.
7. Open a draft PR to `main` after the first local environment + Q1 reproduction commit.

## First expected commit

The first Windows-machine commit should contain at least:

- `docs/experiments/e0/windows_environment.snapshot.json`
- `docs/experiments/e0/LOCAL_BOOTSTRAP_STATUS.md`
- local Q1 audit output or a manifest pointing to it

After pushing, comment on issue #1 with the commit SHA, GPU/RAM summary, and whether Q1 reproduced exactly.
