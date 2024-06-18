# Python Template Repository

This repository hosts the setup scripts and metadata for new OS-Climate Python projects

## Bootstrap Scripts, Templating and Skeleton Files

Raise GitHub issues here if/when you need a new OS-Climate GitHub repository creating

## Description

Repository and CLI tool naming should reflect and match installable package names.
Repository names are prefixed with "osc-" to help avoid wider name-space conflicts.
Repository names use dashes, while module names and some folders may use underscores.

Package names should be generic

**Note:** _this ensures consistency if/when packages are made available through PyPI_

- We use the following tool to bootstrap new projects: [Pyscaffold](https://pyscaffold.org/en/stable/)
- Initial linting and GitHub workflows are imported from: [devops-toolkit](https://github.com/os-climate/devops-toolkit/)

The setup script does the following:

- Invokes pyscaffold to create a folder hierarchy (based on modern PEP standards)
- Creates default linting, TOX and project metadata files
- Performs some post-installation customisation to Pyscaffold (specific to OS-Climate)
- Imports an enhance linting setup from a central OS-Climate reposiutory
- Imports a bunch of shared GitHub actions workflow for common Python project needs

## Modern PEP Standards Compliance

We aim to ensure our projects start with the latest PEP standards compliance in mind

To this end, we do NOT use the following:

- Setuptools (for builds)
- requirements.txt (for describing module dependencies)

Instead we are using the following:

- PDM project (build/dependency management tool)
- pyproject.toml (project metadata description)

### PDM Project

For further details on using PDM for managing dependencies and builds, see:

- [PDM Project](https://pdm-project.org/en/latest/)
- [PDM Project on GitHub](https://github.com/pdm-project/pdm)
- [PDM/Setup Github Action](https://github.com/pdm-project/setup-pdm)

### Information on pyproject.toml

- [Guide to writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [File Specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [Syntax/Cheat Sheet](https://betterprogramming.pub/a-pyproject-toml-developers-cheat-sheet-5782801fb3ed)

<!--
[comment]: # SPDX-License-Identifier: Apache-2.0
[comment]: # Copyright 2024 The Linux Foundation <mwatkins@linuxfoundation.org>
-->
