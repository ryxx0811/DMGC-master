import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Multi-graph Learning Configuration')

    parser.add_argument('--task', choices=['single_cell', 'heart', 'spatial'], default='spatial')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    parser.add_argument('--checkpoint', type=str, default=None, help='model checkpoint for saving or evaluation')
    parser.add_argument('--output_dir', type=str, default='results', help='directory for model outputs')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=3000, help='number of epochs to train')
    parser.add_argument('--print_every', type=int, default=20, help='How often to print training info.')
    parser.add_argument('--emb', type=int, default=100, help='embedding size')
    parser.add_argument('--beta', type=float, default=2.0, help='weight for beta loss')
    parser.add_argument('--sharedW', action=argparse.BooleanOptionalAction, default=False, help='share the W matrix')

    parser.add_argument('--l0', type=float, default=3.3, help='weight for clustering loss')
    parser.add_argument('--l1', type=float, default=0.4, help='weight for inside-graph regularization term')
    parser.add_argument('--l2', type=float, default=5e-4, help='weight for cross-graph regularization term')
    parser.add_argument('--l11', type=int, default=1, help='weight for second order loss')
    parser.add_argument('--l01', type=int, default=1, help='weight for uniform loss')
    parser.add_argument('--centroid_dim', type=int, default=50, help='shared centroid dimmensions')
    parser.add_argument('--center_method', type=str, default='concat', help='center_method')
    parser.add_argument('--hdim', type=int, default=128, help='hidden represenation size')
    return parser.parse_args()
