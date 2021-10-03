Development of a model for recommending title and theme for a news article
==============================

## Motivation

In the last 18 months, many people started joining online groups and used video conferencing for the first time because of the COVID19 effect on our day-to-day life. The digital transformation accelerated in all aspects of people communication, such as using new digital tools. 

While the COVID19 crisis threat has been receding with the introduction of vaccines, it proved that the audience's need for fast access to reliable news is critical for our collective responsibility for longer-term threats such as climate change. Moving forward, publishers increasingly recognize that long-term survival is likely to involve stronger and more profound connections with audiences online. A recent study on the future of digital news indicated that most people are not paying for online news, and publishers rely on the advertisement. For those who subscribe, the most important factor is the distinctiveness and quality of the content.

## Approach

This project will review over 350,000 news articles published in 2020 that were focused on non-medical aspects of the pandemic. The goal is to develop a machine-learning tool to analyze a manuscript, extract the main topics in the document, and suggest appropriate tags and a title for the article. The tags facilitate a more accurate understanding of the main themes in a document and help writers and publishers ensure readers can easily find the content. 

This project is a supervised learning classification problem, and I will use multiple algorithms to compare their performance for the given task. The final product will be accessible via an API and a website. The website interface will provide a window for the authors to write their manuscript and receive feedback on tags and the title.

## Data sources

The main source of data for this project is the `Covid-19 Public Media Dataset by Anacode` on Kaggle.

> The database is a resource of over 350,000 online articles with full texts which were scraped from online media in the time span January 1 - December 31, 2020. The dataset includes articles from more than 60 English-language resources from various domains and is meant to provide a representative cross-section of the online content that was produced and absorbed during this period. The heart of this dataset are online articles in text form - thus, it is meant to be analyzed using Natural Language Processing and text mining. The data was scraped from a range of more than 60 high-impact blogs and news websites. We have been scraping the 65 websites continuosly every day, after which we filtered them on Covid19-related keywords to keep only relevant articles in the dataset. There are 5 topic areas - general, business, finance, tech and science.









## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
