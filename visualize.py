import argparse
from libcity.utils.visualize import VisHelper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='Seattle', help='the name of dataset')
    parser.add_argument('--save_path', type=str,
                        default="./visualized_data/", help='the output path of visualization')

    args = parser.parse_args()

    helper = VisHelper(vars(args))
    helper.visualize()
