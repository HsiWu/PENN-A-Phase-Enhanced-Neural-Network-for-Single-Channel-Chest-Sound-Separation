import argparse
import utils.test as test


def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Model training and testing configuration")

    # Add command-line arguments


    parser.add_argument('--cut_off_heart', type=str, default='250Hz', help='Heart cutoff frequency, default is 250Hz')
    parser.add_argument('--cut_off_lung', type=str, default='60Hz', help='Lung cutoff frequency, default is 60Hz')
    parser.add_argument('--model', type=str, default='Dual_Trans_Decoder', help='Model type, default is Dual_Trans_Decoder, can also be Dual_Trans')
    parser.add_argument('--complex_mask', type=bool, default=False, help='Use complex mask or not, default is False')
    parser.add_argument('--path', type=str, default='Dual', help='Transformer type, default is Dual, can also be Time or Fre')
    parser.add_argument('--pure_phase', type=bool, default=False, help='Use pure phase or not, default is False')
    parser.add_argument('--dual_decoder', type=bool, default=False, help='Use dual decoder or not, default is False')
    parser.add_argument('--root_dir', type=str, default=r'./data', help='Root directory path, default is ./data')
    parser.add_argument('--classify', type=str, default=None, help='Only effective when generating the test dataset. '
                                                                     'Whether divide the test dataset according to pathological condition or not.')
    parser.add_argument('--test_num', type=int, default=2000, help='Number of test samples, default is 2000')

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the test function with the parsed arguments
    test.test(
        root_dir=args.root_dir,
        cut_off_heart=args.cut_off_heart,
        cut_off_lung=args.cut_off_lung,
        model=args.model,
        path=args.path,
        complex_mask=args.complex_mask,
        pure_phase=args.pure_phase,
        dual_decoder=args.dual_decoder,
        test_num=args.test_num,
        classify=args.classify
    )

if __name__ == "__main__":
    main()
