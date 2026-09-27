<div align="center">
	<!--
	<img src="logo.png" alt="Logo">-->
	<h1 style="color: #3081ba">Familiars</h1>
	<div align="center" style="border: 5px solid #2980b9; border-radius: 8px; box-shadow: 0 0 10px rgba(36, 124, 180, 0.5), 0 0 30px 6px rgba(36, 124, 180, 0.5), inset 0 0 5px #00f3ff;">
	<p>A familiar is a magical spirit or creature that,
	according to European folklore and witchcraft traditions,
	serves a witch, sorcerer, or mage, assisting them in sorcery,
	protecting them from enemies, and often acting as a loyal companion.</p>
	</div>
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

### Familiars. Neural AI models for playing some computer games, based on PyTorch and Transformers
###### *[(Go to changelog)](CHANGELOG.md)*

## Table of contents
1. [Installing](#installing)
2. [Links](#links)
3. [Documentation](#documentation)
4. [Build](#build)
5. [Workflows](#workflows)

## Installing

+ Go to https://pytorch.org/projects/pytorch/, select and install torch for your cpu/gpu *(recommended CUDA GPU)*
+ Next, exec:
```bash
python -m pip install familiars-ai
```

**I recommend using `venv` for large libraries like PyTorch and Transformers.**

## Links

+ **GitHub:** https://github.com/IgorNk500/familiars
+ **Documentation:** https://github.com/IgorNk500/familiars/wiki
+ **Issues:** https://github.com/IgorNk500/familiars/issues
+ **Changelog:** https://github.com/IgorNk500/familiars/blob/main/CHANGELOG.md

## Documentation

**See [here](https://github.com/IgorNk500/familiars/wiki)**\
**Docs for screen-familiars see in [repo](https://github.com/IgorNk500/screen-familiars)
or [here](https://github.com/IgorNk500/screen-familiars/wiki)**

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
