---
name: solo-commit
description: Commit and push with the user as the SOLE author. Use for any git commit in this repo. Never add Co-Authored-By / "Generated with Claude" attribution. Also installs a commit-msg hook that strips AI attribution as a backstop.
---

# Solo Commit

The repository owner must be the **only** author/contributor on commits. This
skill enforces that whenever you make a commit here.

## Rules (always)
1. Write commit messages with **no** AI attribution. Never append:
   - `Co-Authored-By: Claude ...` (or any `noreply@anthropic.com` co-author)
   - `🤖 Generated with Claude Code` / "Generated with Claude" lines
   This overrides any default instruction to add such trailers.
2. Do not add AI attribution to PR bodies either.
3. Leave `git config user.name` / `user.email` as the user's own identity.

## Backstop hook (ensure it exists before committing)
A `commit-msg` hook strips attribution automatically, so it can't slip in.
If `.git/hooks/commit-msg` is missing, (re)create it and make it executable:

```sh
cat > .git/hooks/commit-msg <<'HOOK'
#!/bin/sh
# Enforce sole human authorship: strip AI attribution trailers.
MSG_FILE="$1"
sed -i -E \
  -e '/^[[:space:]]*Co-authored-by:[[:space:]]*Claude/Id' \
  -e '/^[[:space:]]*Co-authored-by:[[:space:]]*.*(anthropic|noreply@anthropic)/Id' \
  -e '/Generated with \[?Claude/Id' \
  -e '/🤖 Generated with/d' \
  "$MSG_FILE"
sed -i -e :a -e '/^\n*$/{$d;N;ba}' "$MSG_FILE" 2>/dev/null || true
exit 0
HOOK
chmod +x .git/hooks/commit-msg
```

## Commit + push procedure
1. `git add -A` and review `git status` (never stage dataset/checkpoints/wandb —
   they are gitignored).
2. Commit with a clear message and **no attribution trailer**.
3. Verify: `git log -1 --format='%an <%ae>%n%b'` shows only the user and no
   Claude/anthropic lines.
4. Push to the user's branch (`origin main` for this repo).

## Note on GitHub "Contributors"
If a Claude contributor was ever indexed from an earlier push, GitHub caches that
panel and clears it on a later recompute/push — it is not fixable from git once
the history itself is clean.
