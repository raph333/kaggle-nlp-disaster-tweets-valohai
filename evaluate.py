import argparse
import pandas as pd
from sklearn.model_selection import cross_val_score
from utils.model import MODEL_PIPELINE

import valohai as vh

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate logistic regression model using cross-validation.')
    parser.add_argument('--validation_data', nargs='?', default='data/train.csv',
                        help='path to training-data csv file')
    parser.add_argument('--x_column', nargs='?', default='text',
                        help='name of the text-column in the csv-file')
    parser.add_argument('--y_column', nargs='?', default='target',
                        help='name of the target column csv-file')
    parser.add_argument('--cv_folds', nargs='?', default=5)
    parser.add_argument('--score', nargs='?', default='accuracy')
    args = parser.parse_args()

    df = pd.read_csv(args.validation_data)
    scores = cross_val_score(estimator=MODEL_PIPELINE,
                             X=df[args.x_column],
                             y=df[args.y_column],
                             scoring=args.score,
                             cv=int(args.cv_folds))

    with vh.logger() as logger:
        logger.log(args.score, scores.mean())

