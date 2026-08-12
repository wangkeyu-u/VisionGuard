# Rollback guide

This work started from the repository's original default branch `main` at
commit `5e8769c9aad00ad529b2b3b818252534306d001c`. All implementation commits live
on `codex/interview-alignment`; the original branch is not rewritten and no
changes are pushed by this workflow.

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

## Remove only the local implementation branch

After switching away from it and preserving any desired work:

```bash
git branch -D codex/interview-alignment
```

This deletes the local implementation branch and its commits. It does not
change `main` or any remote branch.
