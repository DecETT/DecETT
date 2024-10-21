import argparse


def arg_parser():

    parser = argparse.ArgumentParser(description='NTA-ML')

    parser.add_argument('--mode', type=str, default='', help='train or test or ""')
    parser.add_argument('--model_file', type=str, default='', help='model file path')
    
    parser.add_argument('--epochs', type=int, default=100, help='raw train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--device', type=str, default='cuda:3', help='device')

    parser.add_argument('--dataset', type=str, default='', help='dataset name')
    parser.add_argument('--path', type=str, default='', help='train/finetune .log file path')
    
    parser.add_argument('--class_num', type=int, help='class num')
    parser.add_argument('--label', type=str, default='label', help='Label column name in .log file')
    parser.add_argument('--max_packet_len', type=int, default=1460, help='max_packet_len')
    
    parser.add_argument('--feature', type=str, default='', help='feature extraction method')
    parser.add_argument('--min_num_pkts', type=int, default=1, help='min_pkt_len')
    parser.add_argument('--max_num_pkts', type=int, default=1000, help='max_pkt_len')
    parser.add_argument('--valid_ratio', type=float, default=0, help='valid ratio')
    parser.add_argument('--test_ratio', type=float, default=1., help='test ratio')
    parser.add_argument('--max_size', type=int, default=0, help='max_size')

    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='loss function')
    parser.add_argument('--recon_loss', action='store_true', default=False, help='use recon loss')

    parser.add_argument('--random_seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose.')

    return parser.parse_args()