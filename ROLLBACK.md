# Rollback guide

This work started from the repository's original default branch `main` at
commit `5e8769c9aad00ad529b2b3b818252534306d001c`. All implementation commits live
on `codex/interview-alignment`; the original branch is not rewritten and no
changes are pushed by this workflow.

Round 1 completed at commit
`4085bb350c193a026aeebe83bcfc5ddf55da1d1e`, protected by the local-only
annotated tag `codex/round1-complete`. Round 2 continues on the same branch.

## Return to the original repository state

Preserve any uncommitted work first, then switch to the recorded baseline:

```bash
git status
git stash push --include-untracked -m "before VisionGuard rollback" # if needed
git switch --detach 5e8769c9aad00ad529b2b3b818252534306d001c
```

To return to the original default branch instead:

```bash
git switch main
git status
git rev-parse HEAD
```

The final command should print
`5e8769c9aad00ad529b2b3b818252534306d001c` as long as `main` has not been
advanced independently.

## Return only to the end of round 1

To inspect the exact round-1 snapshot without moving or rewriting a branch:

```bash
git status
git stash push --include-untracked -m "before VisionGuard round-1 rollback" # if needed
git switch --detach codex/round1-complete
git rev-parse HEAD
```

The final command must print
`4085bb350c193a026aeebe83bcfc5ddf55da1d1e`. To restart the implementation
branch from that snapshot after deliberately discarding round-2 commits, first
switch away from the branch and then run:

```bash
git branch -f codex/interview-alignment codex/round1-complete
git switch codex/interview-alignment
```

The branch-reset command discards round-2 branch reachability, so use it only
after preserving any desired round-2 commits or creating a backup tag/bundle.

## Remove only the local implementation branch

After switching away from it and preserving any desired work:

```bash
git branch -D codex/interview-alignment
```

This deletes the local implementation branch and its commits. It does not
change `main` or any remote branch.
