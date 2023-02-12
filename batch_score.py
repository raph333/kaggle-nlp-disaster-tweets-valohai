import argparse
import pickle
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Add model-prediction to dataset.')
    parser.add_argument('input_dataset', nargs='?', default='data/test.csv')
    parser.add_argument('output_dataset', nargs='?', default='batch_prediction.csv')
    parser.add_argument('x_column', nargs='?', default='text',
                        help='name of the text-column in the csv-file')

    arguments = parser.parse_args()

    with open('model.pkl', 'rb') as infile:
        model = pickle.load(infile)

    df = pd.read_csv(arguments.input_dataset)
    df['prediction'] = model.predict(df[arguments.x_column])
    df.to_csv(arguments.output_dataset, index=False)
