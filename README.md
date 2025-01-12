
# MLOPS Project Description - NLP Sentiment Analysis of iMDB reviews

s224959 Sofus Carstens //
s204701 Lukas Raad //
s225083 Christian Warburg //

## Overall goal for the project

The goal of this project is to solve a binary classification task whether a review is negative or positve.

## Framework for our project

To solve this classification task we plan to use the hugging face Transformers repo. More specifically a pretrained model with a classification head (BERT)

## Dataset (might change)

We are using the IMDb dataset from Hugging Face. Each sample in the dataset consists of a movie review along with a binary sentiment label indicating whether the review is positive (1) or negative (0). The dataset is widely used for sentiment analysis tasks and includes reviews of varying lengths and complexities. The dataset was chosen because it provides a balanced and comprehensive set of text samples for exploring Natural Language Processing techniques. Its straightforward binary classification task makes it well-suited for developing and evaluating machine learning models in a limited timeframe.

## What deep learning models do you expect to use?  
We plan to use **DistilBERT**, a lightweight transformer-based model designed for natural language understanding tasks. DistilBERT is a smaller, faster, and more efficient version of BERT, making it an ideal choice for our sentiment analysis task on the IMDb dataset. Its reduced size allows for quicker training and inference without significant loss in performance, making it feasible to implement within our timeframe while still leveraging the power of pre-trained transformer models.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
