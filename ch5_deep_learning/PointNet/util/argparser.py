import argparse

def parse_arguments():
    """[For user-friendly command-line interfaces]

    Returns:
        [type]: [description]
    """
    parser = argparse.ArgumentParser(description = 'parse arguments')
    
    parser.add_argument('--data_dir',
                        default='/mvtec/home/jinx/privat/modelnet40_normal_resampled',
                        help="Path of dataset.")
    
    parser.add_argument('--exp_dir',
                        default='/mvtec/home/jinx/privat/exp1TestBaseLine',
                        help="Path of exp_dir.")
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=5,
                        help="Path of train dataset.")
    
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help="Num of class you want to classify.")
    
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help="learning_rate.")
    
    parser.add_argument('--num_epochs',
                    type=int,
                    default=1000,
                    help="num_epochs.")
    
    parser.add_argument('--weight_decay',
                    type=float,
                    default=1e-5,
                    help="weight decay.")
    
    
    args = parser.parse_args()
    
    return args
    