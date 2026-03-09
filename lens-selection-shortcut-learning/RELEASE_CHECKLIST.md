# Release checklist

## Pre-release

- [ ] README.md is up to date with current results
- [ ] environment.yml and requirements.txt match working environment
- [ ] Demo script runs and exits cleanly (with or without checkpoint)
- [ ] `scripts/make_submission_figures.sh` runs without errors (or fails gracefully)
- [ ] Key figures are committed in `figures/`
- [ ] CITATION.cff metadata is correct
- [ ] LICENSE file is present
- [ ] .gitignore excludes large files, includes committed figures
- [ ] No secrets, credentials, or absolute paths in committed files
- [ ] CI workflow passes on GitHub

## Release

- [ ] Tag release: `git tag v1.0.0`
- [ ] Create GitHub release from tag
- [ ] Connect repository to Zenodo
- [ ] Obtain DOI from Zenodo

## Post-release

- [ ] Update README.md with Zenodo DOI badge
- [ ] Update CITATION.cff with DOI
- [ ] Update manuscript with repository URL and DOI
- [ ] Verify Zenodo archive contains the correct snapshot
