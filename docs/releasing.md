# Releasing AssetForge

AssetForge is distributed as the `assetforge-2d` Python package. The runtime
is free and local; image-generation providers are optional inputs and are not
downloaded by the package.

## Verify a release locally

Run from the repository root. The project contains a `build/` artifact folder,
so invoke the build frontend from outside the repository to avoid Python
module-name shadowing:

```bash
python -m unittest discover -s assetforge/tests -v
cd /tmp
python -m build /path/to/assetforge --outdir /tmp/assetforge-dist
python -m twine check /tmp/assetforge-dist/*
```

Install the wheel into a clean environment and verify `assetforge --version`,
`assetforge profiles`, and `assetforge pipeline --help`.

## Publish

The GitHub Actions workflow `.github/workflows/publish.yml` publishes only
version tags matching `vMAJOR.MINOR.PATCH`, using PyPI Trusted Publishing (OIDC)
and the GitHub environment named `pypi`.

Before the first tag, create the `pypi` environment and configure the PyPI
project's trusted publisher for this repository and the `publish.yml` workflow.
Then update `CHANGELOG.md`, commit, and push a matching tag:

```bash
git tag v0.6.0
git push origin v0.6.0
```

The workflow builds both sdist and wheel, runs `twine check`, and publishes the
artifacts. A publish is not considered complete until a clean environment can
install the wheel from PyPI and run the CLI smoke commands.
