import argparse
from utils.calculate import calculate

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Parameter configuration for training the model")

    # Add command-line arguments
    parser.add_argument('--cut_off_heart', type=str, default='250Hz', help='Heart cutoff frequency, default is 250Hz')
    parser.add_argument('--cut_off_lung', type=str, default='60Hz', help='Lung cutoff frequency, default is 60Hz')
    parser.add_argument('--model', type=str, default='Dual_Trans_Decoder', help='Model type, default is Dual_Trans_Decoder, can also be Dual_Trans')
    parser.add_argument('--complex_mask', type=bool, default=False, help='Use complex mask or not, default is False')
    parser.add_argument('--path', type=str, default='Dual', help='Transformer type, default is Dual, can also be Time or Fre')
    parser.add_argument('--pure_phase', type=bool, default=False, help='Use pure phase or not, default is False')
    parser.add_argument('--dual_decoder', type=bool, default=False, help='Use dual decoder or not, default is False')


    # Parse command-line arguments
    args = parser.parse_args()

    # Call the train function with the parsed arguments (run it once)
    calculate(
        cut_off_heart=args.cut_off_heart,
        cut_off_lung=args.cut_off_lung,
        model=args.model,
        path=args.path,
        complex_mask=args.complex_mask,
        pure_phase=args.pure_phase,
        dual_decoder=args.dual_decoder,
    )

if __name__ == "__main__":
    main()