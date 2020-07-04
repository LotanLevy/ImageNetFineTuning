import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--nntype', default="PerceptualModel", help='The type of the network')
    parser.add_argument('--cls_num', type=int, default=1000, help='The number of classes in the dataset')
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--last_frozen_layer', '-frozen', type=int, default=19)
    parser.add_argument('--lambd', type=float, default=0.1)



    parser.add_argument('--ref_train_path', type=str, required=True)
    parser.add_argument('--ref_val_path', type=str, required=True)
    parser.add_argument('--ref_test_path', type=str, required=True)
    parser.add_argument('--tar_train_path', type=str, required=True)
    parser.add_argument('--tar_val_path', type=str, required=True)
    parser.add_argument('--tar_test_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')
    parser.add_argument('--print_freq', '-pf', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batchs_num', '-bs', type=int, default=32, help='number of batches')
    parser.add_argument('--train_iterations', '-iter', type=int, default=800, help='The maximum iterations for learning')
    parser.add_argument('--test_layer', type=str, default="fc2")

    parser.add_argument('--test_num', type=int, default=100, help='The number of test examples to consider')
    parser.add_argument('--templates_num', '-tn', type=int, default=40, help='The number pf templates in the testing')




    return parser.parse_args()




