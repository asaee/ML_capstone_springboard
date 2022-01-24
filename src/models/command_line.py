import sys
import logging
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from train_model import *

logging.basicConfig(
    format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)


@click.command()
@click.option('--filename',
              type=click.Path(exists=True),
              prompt='Path to the CSV file',
              help='Path to the CSV file')
def articles_analysis(filename):
    train_classifier(filename)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    articles_analysis()
