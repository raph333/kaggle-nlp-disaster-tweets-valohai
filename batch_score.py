import argparse
import pickle
import pandas as pd
import uuid
import valohai as vh

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Add model-prediction to dataset.')
    parser.add_argument('--data', nargs='?', default='data/test.csv',
                        help='path to data-set')
    parser.add_argument('--x_column', nargs='?', default='text',
                        help='name of the text-column in the csv-file')
    parser.add_argument('--model', nargs='?',
                        help='for local runs: path to local model-file')
    args = parser.parse_args()

    # get model argument only for local runs - otherwise from valohai inputs
    model_path = args.model or vh.inputs('model').path()

    with open(model_path, 'rb') as infile:
        model = pickle.load(infile)

    df = pd.read_csv(args.data)
    df['prediction'] = model.predict(df[args.x_column])

    output_path = vh.outputs('model').path(f'predictions-{uuid.uuid4()}.csv')
    df.to_csv(output_path, index=False)
