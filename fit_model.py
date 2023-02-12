import argparse
import pickle

import pandas as pd

from utils.model import MODEL_PIPELINE


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train logistic regression model')
    parser.add_argument('training_data', nargs='?', default='data/train.csv',
                        help='path to training-data csv file')
    parser.add_argument('x_column', nargs='?', default='text',
                        help='name of the text-column in the csv-file')
    parser.add_argument('y_column', nargs='?', default='target',
                        help='name of the target column csv-file')

    arguments = parser.parse_args()

    df = pd.read_csv(arguments.training_data)
    MODEL_PIPELINE.fit(df[arguments.x_column], df[arguments.y_column])
    with open('model.pkl', 'wb') as outfile:
        pickle.dump(MODEL_PIPELINE, outfile)
