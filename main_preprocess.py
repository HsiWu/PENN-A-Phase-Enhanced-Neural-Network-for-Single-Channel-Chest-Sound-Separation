import argparse
from utils.preprocess import preprocess

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Parameter configuration for training the model")

    # Add command-line arguments
    parser.add_argument('--database_path', type=str, default='./data', help='The original dataset path, default is ./data')
    parser.add_argument('--output_path', type=str, default='./data', help='The dataset path after preprocess, default is ./data')
    parser.add_argument('--signal_type', type=str, required=True, help='signal type, be either heart or lung')
    parser.add_argument('--fc', type=int, default=None, help='cutoff frequency, default is None. '
                                                             'At this time, fc=250 if signal is heart and fc=60 if signal is lung.'
                                                             'Can also be specified.')
    parser.add_argument('--train', type=bool, required=True, help='Generate the training or testing dataset.')
    parser.add_argument('--classify', type=bool, default=False, help='Only effective when generating the test dataset. '
                                                                     'Whether divide the test dataset according to pathological condition or not.')


    # Parse command-line arguments
    args = parser.parse_args()

    # Call the train function with the parsed arguments (run it once)
    preprocess(
        database_path=args.database_path,
        output_path=args.output_path,
        signal_type=args.signal_type,
        fc=args.fc,
        train=args.train,
        classify=args.classify,
    )

if __name__ == "__main__":
    main()