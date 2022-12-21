# Repo

Short Description


## Setting up the environment

[Anaconda](https://www.anaconda.com/) is required to create virtual environments.

- Create new environments for the project

  ```
  make create_environment
  ```

- Activate created or existing environment

  ```
  conda activate ./env
  ```

## Documentation

For an in-depth documentation of the commands.

- Run: `make documentation`
- Open: `docs/_build/html/index.html`

## Commands

The `Makefile` and `setup.py` contains the central entry points for common tasks related to this project.

- Makefile commands: Run `make help` to list all available commands.
- CLI commands: Run each command to get a list of its sub-commands.

| Command         | Description             |
| --------------- | ----------------------- |
| `pipeline`      | Run entire pipeline.    |
| `data`          | Process raw data.       |
| `features`      | Pre-process raw data.   |
| `models`        | Run inference.          |
| `visualization` | Visualize results.      |

