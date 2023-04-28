# Repo

This is currently becoming something

## Setting up the environment

[Anaconda](https://www.anaconda.com/) is required to create dev environment.

- Create new environment for the project

  ```
  make create_environment
  make requirements
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

| Command      | Description         |
| ------------ | ------------------- |
| `pipeline`   | Run entire pipeline |
| `world`      | Visualize world     |
| `adversary`  | Adversary tests     |
| `player`     | Player tests        |

- `adversary train` to run the training script
-  logging is done for each ray worker

### Common commands

```
adversary train -v -d "6x10 test"
player test ./models/example.pth -s random -v
results record ./logs/XXX_policy.pth -s evasive
```

# Windows Guide

```
conda env create --prefix ./$(ENV_DIR) --file environment.yml
conda activate ./env
pip install --no-deps -r requirements_freezed_linux_0304.txt
```

# TODO

- [x] Log episode rewards
- [x] Saving and loading models from file
- [ ] Player styles
- [ ] Evaluation
- [ ] Fix order of rewards
