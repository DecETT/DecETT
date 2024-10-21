# encoding:utf-8
import os
import time

import torch
from loguru import logger as logging
from torch.utils.data import DataLoader

from eval import eval_results
from test import test
from params import arg_parser
from dataset import Dataset, DatasetProcessor_DRL
from features import seq_features

MAX_PACKET_LEN = 3000

def initialize(args):

    work_dir = os.path.abspath(os.path.dirname(__file__))
    
    work_time = time.strftime(f"%y-%m-%d_%H-%M-%S", time.localtime())
    
    task = args.mode if args.mode != '' else 'train-test'
    results_dir = os.path.join(work_dir, 'saved_results', f'{work_time}_{args.model}_{args.dataset}_{task}')
    os.makedirs(results_dir, exist_ok=True)

    log_file = '.'.join(os.path.basename(args.path).split(".")[:-1])
    data_dir = os.path.join(work_dir, 'saved_data', args.dataset, log_file)
    os.makedirs(data_dir, exist_ok=True)

    if not args.verbose:
        logging.remove(handler_id=0)
    logid = logging.add(os.path.join(results_dir, 'runtime.log'), level='DEBUG')

    logging.info(f'Work dir: {work_dir}')
    logging.info(f'Work time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

    model_dir = os.path.join(work_dir, 'saved_models')
    os.makedirs(os.path.join(work_dir, 'saved_models'), exist_ok=True)

    return logid, data_dir, results_dir, model_dir, work_time



if __name__ == "__main__":
    
    args = arg_parser()

    logid, data_dir, results_dir, model_dir, work_time = initialize(args)

    feature_cls = seq_features.DRLPacketSizeSequence(max_len=200, signed=True)

    logging.info(f'Task info: Feature extractor: {feature_cls.__class__.__name__}; Model: DecETT; Loss: DecETT_Loss')

    logging.info('Loading dataset...')

    processor = DatasetProcessor_DRL(args, feature_extractor=feature_cls)


    if os.path.exists(os.path.join(data_dir, 'data.pkl')):
        X, y, S, preprocess_time, label_dict = processor.load(os.path.join(data_dir, 'data.pkl'))
        logging.info(f'Load stored data success! Time cost: {preprocess_time:.2f}s')
    else:
        X, y, S, preprocess_time, label_dict = processor.process()
        processor.save(os.path.join(data_dir, 'data.pkl'), X, y, S, label_dict)
        logging.info(f'Preprocess data success! Time cost: {preprocess_time:.2f}s')


    if args.mode == '':
        split_data = processor.split(X, y, S, args.valid_ratio, args.test_ratio, args.random_seed)
        X_train, X_valid, X_test, y_train, y_valid, y_test, S_train, S_valid, S_test = split_data
    elif args.mode == 'train':
        split_data = processor.split(X, y, S, args.valid_ratio, 0., args.random_seed)
        X_train, X_valid, _, y_train, y_valid, _, S_train, S_valid, S_test = split_data
    elif args.mode == 'test':
        split_data = processor.split(X, y, S, args.valid_ratio, args.test_ratio, args.random_seed)
        X_train, X_valid, X_test, y_train, y_valid, y_test, S_train, S_valid, S_test = split_data


    logging.info(f'Split data success!')

    if args.mode == '' or args.mode == 'test':
        logging.info('----------- Testing -----------')

        t_start_model_load = time.time()
        model = torch.load(os.path.join(model_dir, args.model_file), map_location=torch.device(args.device))
        t_end_model_load = time.time()
        t_load = t_end_model_load - t_start_model_load
        logging.info(f'Load model success! Path: {os.path.join(model_dir, args.model_file)}')

        test_loader = DataLoader(dataset=Dataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        y_pred, y_pred_proba, t_pred = test(model, test_loader, args)
    
        logging.info('----------- Evaluation -----------')
        test_result_file = os.path.join(results_dir, f'{args.feature}_{args.model}_{args.dataset}_results_test.json')
        eval_results(f'{args.feature}_{args.model}_{args.dataset}', 
                    y_test, y_pred, y_pred_proba,
                    argparser_params=vars(args),
                    model_load_time=t_load, 
                    test_samples=len(y_test),
                    test_time=t_pred,
                    out=test_result_file)

    logging.info('Finish!\n')