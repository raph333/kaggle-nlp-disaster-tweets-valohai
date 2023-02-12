import argparse
import pandas as pd
from sklearn.model_selection import cross_val_score
from utils.model import MODEL_PIPELINE

import valohai

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='evaluate logistic regression model')
    parser.add_argument('--validation_data', nargs='?', default='data/train.csv',
                        help='path to training-data csv file')
    parser.add_argument('--x_column', nargs='?', default='text',
                        help='name of the text-column in the csv-file')
    parser.add_argument('--y_column', nargs='?', default='target',
                        help='name of the target column csv-file')
    parser.add_argument('--cv_folds', nargs='?', default=5)
    parser.add_argument('--score', nargs='?', default='accuracy')
    arguments = parser.parse_args()

    df = pd.read_csv(valohai.parameters('validation_data').value)
    scores = cross_val_score(estimator=MODEL_PIPELINE,
                             X=df[valohai.parameters('x_column').value],
                             y=df[valohai.parameters('y_column').value],
                             scoring=valohai.parameters('score').value,
                             cv=valohai.parameters('cv_folds').value)

    with valohai.logger() as logger:
        logger.log(valohai.parameters('score').value, scores.mean())

