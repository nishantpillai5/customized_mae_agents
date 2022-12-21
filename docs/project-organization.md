# Project Organization

    ├── data
    │   ├── external               <- Data from third party sources.
    │   ├── interim                <- Intermediate data that has been transformed.
    │   ├── processed              <- The final, canonical data sets for modeling.
    │   └── raw                    <- The original, immutable data dump.
    │
    ├── docs                       <- Sphinx documentation.
    │
    ├── env                        <- Python virtual (conda) environment.
    |
    ├── external repos             <- External repositories for state-of-the-art models.
    │
    ├── logs                       <- Log files.
    │
    ├── models                     <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                  <- Jupyter notebooks.
    │
    ├── references                 <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                    <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                <- Generated graphics and figures to be used in reporting
    │
    ├── Makefile                   <- Makefile with commands like `make data` or `make train`
    ├── download_models.sh         <- Bash script to download pre-trained models.
    │
    ├── environment.yml            <- Conda requirements file for reproducing the environment.
    ├── requirements.txt           <- Generalized requirements file for reproducing the environment.
    ├── requirements_freezed.txt   <- Locked requirements file generated with `pip freeze > requirements.txt`.
    │
    ├── setup.py                   <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── README.md                  <- The top-level README for developers using this project.
    │
    └── src                        <- Source code for use in this project.
        ├── __init__.py            <- Makes src a Python module
        │
        ├── data                   <- Scripts to generate data
        │
        ├── features               <- Scripts to turn raw data into features for modeling
        │
        ├── models                 <- Scripts to train models and then use trained models to make
        │                             predictions
        │
        └── visualization          <- Scripts to create exploratory and results oriented visualizations

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
