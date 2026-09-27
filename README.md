<div align="center">
	<!--
	<img src="logo.png" alt="Logo">-->
	<h1>Familiars</h1>
	<br/>
	<br/>
	<a href="https://github.com/IgorNk500/familiars/actions/workflows/release_build.yml"><img src="https://github.com/IgorNk500/familiars/actions/workflows/release_build.yml/badge.svg" alt="Build"></a>
	<a href="https://github.com/IgorNk500/familiars/actions/workflows/pylint.yml"><img src="https://github.com/IgorNk500/familiars/actions/workflows/pylint.yml/badge.svg" alt="Pylint"></a>
	<a href="https://github.com/IgorNk500/familiars/actions/workflows/pytest.yml"><img src="https://github.com/IgorNk500/familiars/actions/workflows/pytest.yml/badge.svg" alt="Pytest"></a>
	<br/>
	<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/familiars-ai" alt="Supported python versions"></a>
	<a href="https://github.com/IgorNk500/familiars"><img src="https://img.shields.io/badge/github-repo-blue?logo=github" alt="Github repo"></a>

</div>

***
### Python library for working with familiars AI based on pytorch and transformers
###### *[(Go to changelog)](CHANGELOG.md)*

## Table of contents
1. [Installing](#installing)
2. ...
3. [Build](#build)
4. [Workflows](#workflows)

## Installing
```bash
python -m pip install familiars-ai
```

...

## Build
### To build the package, run:
```bash
python -m pip install build
python -m build
```
Don't forget to change the config in the `pyproject.toml` file before doing this.\
**All distributions are stored in the `dist` folder**

## Workflows
**This project contains one workflow that:**
+ collects the wheel library,
+ adds it to the release files,
+ and uploads the release to PyPi.

**Also, project contains pytest and pylint workflows**

## If you encounter any errors, please open [issue](https://github.com/IgorNk500/familiars/issues/new "New issue") on GitHub.
