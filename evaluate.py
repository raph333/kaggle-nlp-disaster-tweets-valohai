import argparse
import pandas as pd
from sklearn.model_selection import cross_val_score
from utils.model import MODEL_PIPELINE

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='evaluate logistic regression model')
    parser.add_argument('validation_data', nargs='?', default='data/train.csv',
                        help='path to training-data csv file')
    parser.add_argument('x_column', nargs='?', default='text',
                        help='name of the text-column in the csv-file')
    parser.add_argument('y_column', nargs='?', default='target',
                        help='name of the target column csv-file')
    parser.add_argument('--cross-validate', '-cv', default=True, action='store_true',
                        help='Use cross-validation if true. Only set it to false if you have a validation set that is '
                             'separate from the training-set.')

    arguments = parser.parse_args()

    df = pd.read_csv(arguments.validation_data)
    scores = cross_val_score(estimator=MODEL_PIPELINE,
                             X=df[arguments.x_column],
                             y=df[arguments.y_column],
                             scoring='f1',
                             cv=5)

    print(scores.mean())
