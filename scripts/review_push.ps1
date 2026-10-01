<#
.SYNOPSIS
    Read-only pre-push review for the Methanol-Catalysis thesis repo.

.DESCRIPTION
    Prints exactly six facts about what a `git push origin main` would publish,
    plus the leak already published in `origin/main` (which no working-tree check
    can see, because `HEAD == origin/main` makes the outgoing range empty).
    It never writes, never stages, never pushes and never fetches. Every git
    invocation is checked: if one exits non-zero the script FAILS CLOSED with
    "BLOCKED: cannot determine what would be published" and a non-zero exit code.

    Fail-closed is the whole point. A review script that degrades to a green
    light when git misbehaves is worse than no review at all: it converts an
    unknown into an apparent approval. "I could not tell" must never render as
    "looks fine".

    Read-only also means it does NOT refresh origin/main. The remote-tracking
    ref is whatever the last fetch left behind and may be stale, so its age is
    reported explicitly rather than being presented as the live remote state.

.NOTES
    Requires PowerShell 5.1 or later. Compatible with both `powershell` and `pwsh`.
#>

$ErrorActionPreference = 'Continue'
$repoName = Split-Path -Leaf (Get-Location).Path

function Write-Fact([string]$label, [string[]]$lines) {
    Write-Output ""
    Write-Output "$label"
    if ($null -eq $lines -or $lines.Count -eq 0) {
        Write-Output "  (empty)"
    }
    else {
        foreach ($line in $lines) { Write-Output "  $line" }
    }
}

function Block([string]$reason, [string]$detail) {
    Write-Output ""
    Write-Output "BLOCKED: cannot determine what would be published"
    Write-Output "reason: $detail"
    Write-Output ""
    Write-Output "BLOCKED: $reason"
    exit 1
}

# --- Preconditions, each with a distinct message ----------------------------

# Remote configured at all?
$remotes = & git remote 2>&1
if ($LASTEXITCODE -ne 0) { Block "no remote configured" "git remote exited $LASTEXITCODE" }
if (-not ($remotes -contains 'origin')) {
    Block "no 'origin' remote configured" ("git remote reports: " + (($remotes | Out-String).Trim()))
}

# Detached HEAD? A detached HEAD is publishable but not by branch name.
$headRef = & git symbolic-ref --quiet --short HEAD 2>&1
if ($LASTEXITCODE -ne 0) {
    Block "detached HEAD" "HEAD is not a symbolic ref; 'git push origin main' would not push what HEAD points at"
}
if (($headRef | Out-String).Trim() -ne 'main') {
    Block "HEAD is on branch '$(($headRef | Out-String).Trim())', not 'main'" "deploy.yml only publishes pushes to main"
}

# Shallow clone? A shallow history cannot answer "what would be published".
$shallow = & git rev-parse --is-shallow-repository 2>&1
if ($LASTEXITCODE -ne 0) { Block "shallow state undetectable" "git rev-parse --is-shallow-repository exited $LASTEXITCODE" }
if ((($shallow | Out-String).Trim()) -eq 'true') {
    Block "shallow clone" "git log/diff over origin/main..HEAD is unreliable without full history"
}

# origin/main present? Without it the outgoing range is undefined.
& git rev-parse --verify --quiet origin/main *> $null
if ($LASTEXITCODE -ne 0) {
    Block "origin/main missing" "no origin/main ref; run 'git fetch origin main' manually, then re-run this review"
}

# origin/main an ancestor of HEAD? A divergence means the range below is not
# a simple "these N commits" story and must not be reported as one.
& git merge-base --is-ancestor origin/main HEAD 2>&1
if ($LASTEXITCODE -ne 0) {
    Block "origin/main diverged from HEAD" "origin/main is not an ancestor of HEAD; origin/main..HEAD is not a clean outgoing set"
}

# --- The six facts ----------------------------------------------------------

Write-Output "PUSH REVIEW: $repoName (read-only; nothing written, staged, fetched or pushed)"

# (a) working tree
$status = & git status --porcelain 2>&1
if ($LASTEXITCODE -ne 0) { Block "git status failed" "git status --porcelain exited $LASTEXITCODE" }
$dirty = @($status).Count -gt 0
$dirtyPaths = @($status | ForEach-Object { $_.Substring(3) })
Write-Fact "(a) WORKING TREE: $(if ($dirty) { 'DIRTY' } else { 'clean' })" @(
    "git status --porcelain ->",
    ($status | Out-String).TrimEnd()
)
if ($dirty) {
    Write-Output "    NOTE: only COMMITTED content publishes. These $(@($dirtyPaths).Count) path(s) are NOT in the push unless you commit them first:"
    foreach ($p in $dirtyPaths) { Write-Output "      - $p" }
}

# (b) outgoing commits, by subject
$log = & git log --oneline origin/main..HEAD 2>&1
if ($LASTEXITCODE -ne 0) { Block "git log failed" "git log --oneline origin/main..HEAD exited $LASTEXITCODE" }
$subjects = @($log | Where-Object { $_ -notmatch '^\s*$' })
Write-Fact "(b) OUTGOING COMMITS (origin/main..HEAD) - $(@($subjects).Count) commit(s), subjects:" $subjects

# (c) outgoing diffstat
$stat = & git diff --stat origin/main..HEAD 2>&1
if ($LASTEXITCODE -ne 0) { Block "git diff --stat failed" "git diff --stat origin/main..HEAD exited $LASTEXITCODE" }
Write-Fact "(c) OUTGOING DIFFSTAT (committed only):" @($stat | Where-Object { $_ -notmatch '^\s*$' })

# (d) references.bib specifically
$bibChanged = & git diff --name-only origin/main..HEAD -- references.bib 2>&1
if ($LASTEXITCODE -ne 0) { Block "git diff --name-only failed" "filtering to references.bib exited $LASTEXITCODE" }
$bibInRange = @($bibChanged | Where-Object { $_ -match '\S' }).Count -gt 0
$bibFieldCount = 0
if (Test-Path -LiteralPath 'references.bib') {
    $bibFieldCount = @(Select-String -LiteralPath 'references.bib' -Pattern '^\s*file\s*=' -AllMatches).Count
}
Write-Fact "(d) references.bib:" @(
    "changed in outgoing range (origin/main..HEAD): $(if ($bibInRange) { 'YES' } else { 'no' })",
    "'file = {' fields in WORKING-TREE copy: $bibFieldCount",
    "note: a non-zero count means Zotero re-exported absolute local paths into the staged-to-be file."
)

# (e) references.bib as ALREADY PUBLISHED, not as it is in the working tree
#
# Fact (d) inspects the working tree and the outgoing range. When HEAD ==
# origin/main -- the normal state right before a first push of local work -- the
# outgoing range is empty, so (d) legitimately reports "changed: no / 0 fields"
# while the file on the public remote still carries the leak. The only thing
# that closes that gap is reading the committed blob itself.
$committedBib = & git show origin/main:references.bib 2>&1
if ($LASTEXITCODE -ne 0) { Block "git show failed" "git show origin/main:references.bib exited $LASTEXITCODE" }
$committedLeak = @(($committedBib -join "`n") -split "`n" |
    Where-Object { $_ -match '^\s*file\s*=' }).Count
$committedLeakLines = @("'file = {' fields in the PUBLISHED copy: $committedLeak")
if ($committedLeak -gt 0) {
    $committedLeakLines += "WARNING: the published bibliography already discloses $committedLeak absolute local path(s)."
    $committedLeakLines += "This push does NOT remove them; they stay in the published history regardless."
    $committedLeakLines += "Rewrite-vs-accept is a SEPARATE, OPEN owner decision (R-PUBLIC), not this script's call."
} else {
    $committedLeakLines += "No 'file' field in the published copy."
}
Write-Fact "(e) PUBLISHED references.bib (read from origin/main, not the working tree):" $committedLeakLines

# (f) the exact SHA that would be published
$sha = & git rev-parse HEAD 2>&1
if ($LASTEXITCODE -ne 0) { Block "git rev-parse failed" "git rev-parse HEAD exited $LASTEXITCODE" }
Write-Fact "(f) HEAD SHA THAT WOULD BE PUBLISHED:" @(($sha | Out-String).Trim())

# --- Executed-on-the-runner warning -----------------------------------------

$ipynbDiff = & git diff --name-only origin/main..HEAD -- '*.ipynb' 2>&1
if ($LASTEXITCODE -ne 0) { Block "git diff --name-only failed" "filtering to '*.ipynb' exited $LASTEXITCODE" }
$ipynbCommitted = @($ipynbDiff | Where-Object { $_ -match '\S' -and $_ -notmatch 'fatal|error' })
$ipynbDirty = @($dirtyPaths | Where-Object { $_ -like '*.ipynb' })
if ($ipynbCommitted.Count -gt 0 -or $ipynbDirty.Count -gt 0) {
    Write-Output ""
    Write-Output "WARNING: notebook changed (committed: $($ipynbCommitted.Count), uncommitted: $($ipynbDirty.Count))."
    Write-Output "  .github/workflows/deploy.yml:89 runs 'myst build --html --execute', so the published"
    Write-Output "  artefact is REBUILT AND RE-EXECUTED on the runner. The HTML you reviewed locally is"
    Write-Output "  not the HTML that ships: notebook cells re-run against the runner environment and can"
    Write-Output "  produce different numbers, or fail, with no local preview of it."
    foreach ($p in @($ipynbCommitted + $ipynbDirty)) { Write-Output "  - $p" }
}

Write-Output ""
Write-Output "PUSH MANUALLY: git push origin main"
exit 0
