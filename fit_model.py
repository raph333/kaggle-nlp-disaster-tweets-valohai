import argparse
import pickle
import pandas as pd
import uuid
import valohai as vh

from utils.model import MODEL_PIPELINE


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fit logistic regression model and save the model-file.')
    parser.add_argument('--training_data', nargs='?', default='data/train.csv',
                        help='path to training-data csv file')
    parser.add_argument('--x_column', nargs='?', default='text',
                        help='name of the text-column in the csv-file')
    parser.add_argument('--y_column', nargs='?', default='target',
                        help='name of the target column csv-file')
    args = parser.parse_args()

    df = pd.read_csv(args.training_data)
    MODEL_PIPELINE.fit(df.text, df.target)

    output_path = vh.outputs('model').path(f'model-{uuid.uuid4()}.pkl')
    with open(output_path, 'wb') as outfile:
        pickle.dump(MODEL_PIPELINE, outfile)
