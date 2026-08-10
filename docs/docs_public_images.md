# Managing docs/public Images

We only want the **latest** version of images in `docs/public/` in the repo—old copies bloat git history and `.git` size.

## Options

### 1. One-time history cleanup (recommended first step)

Remove `docs/public/` from **all** past commits, then add the current directory in a single commit. After this, history contains only one snapshot of those files and `.git` shrinks.

**Requirements:** [git-filter-repo](https://github.com/newren/git-filter-repo) (preferred) or BFG Repo-Cleaner.

```bash
# Install git-filter-repo (e.g. with pip)
pip install git-filter-repo

# Backup and run from repo root
cd /path/to/MillionTrees
cp -r docs/public /tmp/docs_public_backup

git filter-repo --path docs/public --invert-paths --force

# Restore current docs/public and commit
mv /tmp/docs_public_backup docs/public
git add docs/public/
git commit -m "docs: add current docs/public images (history cleaned)"
```

**Warning:** This rewrites history. Everyone who has cloned must re-clone or rebase; force-push is required (`git push --force-with-lease`). Coordinate with collaborators.

### 2. Going forward: Git LFS for new image commits

So future updates to images don’t add huge blobs to the main repo, binary assets in `docs/public/` are tracked via **Git LFS**. The repo already has `.gitattributes` for `docs/public/*.png` (and .jpg, .jpeg, .webp).

One-time on each clone (and for CI if you run git there):

```bash
git lfs install
```

Then add and commit as usual; matching files in `docs/public/` will be stored in LFS. LFS stores file contents separately; the repo keeps only small pointer files. (Old history from before LFS still contains the old blobs until you do the one-time cleanup above.)

### 3. Alternative: Don’t track docs/public in git

If images can be generated (e.g. by `data_prep/package_datasets.py`) or hosted elsewhere:

- Add `docs/public/` to `.gitignore`
- Run `git rm -r --cached docs/public` and commit
- Document in README or this doc that images are generated or stored outside the repo

Then the repo never stores those binaries; downside is new clones don’t get the images unless they run the script or fetch from elsewhere.

## Recommendation

1. Do **Option 1** once to shrink the repo and keep only the latest `docs/public` in history.
2. Add **Option 2** (LFS for `docs/public/*.png` etc.) so future image updates don’t bloat the repo again.
