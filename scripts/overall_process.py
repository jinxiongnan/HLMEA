import json
import os

import torch
from torch.optim import AdamW, Adam

from data_clean import split_data
from dataloader import create_dataloader
from ea_by_llm import ea_by_llm_recursively
from evaluation import evaluate
from train import CustomMPNetModel, train, CustomE5Model, CustomMinilmModel, CustomLabseModel
from training_data_generation import generate_training_data_from_distribution
from utility import compute_llm_output_distribution, move_to_processed_dir
import argparse


def get_ptm_model_name_with_round(ptm_model_name: str, round: int):
    if round != 0:
        ptm_model_name = '%s-round-%d' % (ptm_model_name, round)

    return ptm_model_name


def get_raw_train_dev_data_path(dataset_name: str, llm_mode: str, ea_data_mode: str,
                                triple_sel: str, exp_sel: str, tar_sel: str, old_ptm_model: str,
                                llm_type: str, llm_temp: str, max_rep: int,
                                lower_bound: int, upper_bound: int, train_percentage: float):
    train_bound = int(upper_bound * train_percentage)
    raw_data_path = os.path.join(
        os.getcwd(), '..', 'output', 'training_data', dataset_name,
        'training_data_distribution=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' %
        (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel, old_ptm_model,
         llm_type, llm_temp, max_rep, lower_bound, train_bound))

    train_data_path = os.path.join(
        os.getcwd(), '..', 'output', 'training_data', dataset_name,
        'split_train_test=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' %
        (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel, old_ptm_model,
         llm_type, llm_temp, max_rep, lower_bound, train_bound))
    dev_data_path = os.path.join(
        os.getcwd(), '..', 'output', 'training_data', dataset_name,
        'split_dev_test=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' %
        (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel, old_ptm_model,
         llm_type, llm_temp, max_rep, lower_bound, train_bound))

    return raw_data_path, train_data_path, dev_data_path


def train_ptm_model(train_data_path: str, dev_data_path: str, old_ptm_model_name: str, new_ptm_model_name: str,
                    dataset_name: str, llm_mode: str, ea_data_mode: str, triple_sel: str, exp_sel: str, tar_sel: str,
                    llm_type: str, llm_temp: str, max_rep: int, lower_bound: int, train_bound: int):
    output_model_dir = os.path.join(os.getcwd(), '..', 'output', 'ptm_model', dataset_name)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    output_model_path = os.path.join(output_model_dir, 'ptm=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.pth' %
                                     (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel, new_ptm_model_name,
                                      llm_type, llm_temp, max_rep, lower_bound, train_bound))

    ## if model file already exists, skip training
    if os.path.exists(output_model_path):
        print('Model file already exists ... %s' % output_model_path)
        return 0

    print('Training ptm_model ... %s' % output_model_path)
    input_model_path = os.path.join(os.getcwd(), '..', 'output', 'ptm_model', dataset_name,
                                    'ptm=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.pth' %
                                    (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel,
                                     old_ptm_model_name, llm_type, llm_temp, max_rep, lower_bound, train_bound))

    with open(train_data_path, 'r') as f:
        train_data = json.load(f)

    if 'e5' in old_ptm_model_name:
        train_batch_size = 2
    elif 'labse' in old_ptm_model_name:
        train_batch_size = 4
    elif 'mpnet' in old_ptm_model_name:
        train_batch_size = 8
    elif 'minilm' in old_ptm_model_name:
        train_batch_size = 8
    else:
        raise RuntimeError('Unknown old_ptm_model: %s' % old_ptm_model_name)

    if dev_data_path:
        with open(dev_data_path, 'r') as f:
            dev_data = json.load(f)
    else:
        dev_data = {k: train_data[k] for k in list(train_data)[:train_batch_size]}

    train_dataloader = create_dataloader(train_data, train_batch_size)
    dev_dataloader = create_dataloader(dev_data, train_batch_size)
    device = torch.device(
        "cuda:2" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")

    # load ptm model
    if 'e5' in old_ptm_model_name:
        old_ptm_model = CustomE5Model()
    elif 'labse' in old_ptm_model_name:
        old_ptm_model = CustomLabseModel()
    elif 'mpnet' in old_ptm_model_name:
        old_ptm_model = CustomMPNetModel()
    elif 'minilm' in old_ptm_model_name:
        old_ptm_model = CustomMinilmModel()
    else:
        raise RuntimeError('Unknown old_ptm_model: %s' % old_ptm_model_name)
    if input_model_path and 'round' in input_model_path:
        old_ptm_model.load_state_dict(torch.load(input_model_path, map_location=device))
        print(f"Loaded model from {input_model_path}")

    optimizer = AdamW(old_ptm_model.parameters(), lr=1e-5)

    train(old_ptm_model_name, old_ptm_model, train_dataloader, dev_dataloader, optimizer, device,
          save_path=output_model_path)

    print('Generated ptm model file ... %s' % output_model_path)

    return 0


def parse_dataset_name(abbr_dataset_name: str):
    if abbr_dataset_name == 'dze':
        full_dataset_name = 'DBP15K_ZH_EN'
    elif abbr_dataset_name == 'dje':
        full_dataset_name = 'DBP15K_JA_EN'
    elif abbr_dataset_name == 'dfe':
        full_dataset_name = 'DBP15K_FR_EN'
    elif abbr_dataset_name == 'dw':
        full_dataset_name = 'DW15K_V1'
    elif abbr_dataset_name == 'dy':
        full_dataset_name = 'DY15K_V1'
    elif abbr_dataset_name == 'ddev':
        full_dataset_name = 'DBP15K_DE_EN_V1'
    elif abbr_dataset_name == 'dfev':
        full_dataset_name = 'DBP15K_FR_EN_V1'
    elif abbr_dataset_name == 'ddev100k':
        full_dataset_name = 'DBP100K_DE_EN_V1'
    elif abbr_dataset_name == 'dfev100k':
        full_dataset_name = 'DBP100K_FR_EN_V1'
    else:
        raise RuntimeError('Unsupported abbr_dataset_name: %s' % abbr_dataset_name)

    return full_dataset_name


def parse_ea_data_mode(abbr_ea_data_mode: str):
    if abbr_ea_data_mode == 'train':
        full_ea_data_mode = 'train-20'
    elif abbr_ea_data_mode == 'test':
        full_ea_data_mode = 'test-80'
    else:
        raise RuntimeError('Unsupported abbr_ea_data_mode: %s' % abbr_ea_data_mode)

    return full_ea_data_mode


def main(dataset_name: str, ptm_model_name: str, llm_type: str, ea_data_mode: str, round: int,
         triple_sel: str, tar_sel: str, debug_mode: bool, llm_mode: str, max_rep: int, exp_sel: str,
         llm_temp: str, train_percentage: float):
    # complete dataset_name and ea_data_mode
    dataset_name = parse_dataset_name(dataset_name)
    ea_data_mode = parse_ea_data_mode(ea_data_mode)
    # set lower_bound and upper_bound
    if ea_data_mode == 'train-20':
        lower_bound = 0
        if '15K' in dataset_name:
            upper_bound = 3000
        elif '100K' in dataset_name:
            upper_bound = 20000
        else:
            raise RuntimeError('Unsupported dataset_name: %s' % dataset_name)
    elif ea_data_mode == 'test-80':
        if '15K' in dataset_name:
            lower_bound = 3000
            upper_bound = 15000
        elif '100K' in dataset_name:
            lower_bound = 20000
            upper_bound = 100000
        else:
            raise RuntimeError('Unsupported dataset_name: %s' % dataset_name)
    else:
        raise RuntimeError('Unsupported ea_data_mode: %s' % ea_data_mode)
    # set llm_model
    if llm_type == 'gpt3.5':
        llm_model = 'gpt-3.5-turbo-1106'
    elif llm_type == 'gpt4':
        llm_model = 'gpt-4-turbo-2024-04-09'
    else:
        llm_model = ''

    train_bound = int(upper_bound * train_percentage)
    if round == 0 or ea_data_mode.startswith('test'):
        if ea_data_mode.startswith('test'):
            ptm_model_name = get_ptm_model_name_with_round(ptm_model_name, round)
        print('Overall process status: (1/3)')
        ea_by_llm_recursively(
            dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
            ptm_model_name, llm_type, llm_model, llm_temp, max_rep, lower_bound, upper_bound, train_percentage
        )
        print('Overall process status: (2/3)')
        move_to_processed_dir(dataset_name, llm_mode, ea_data_mode, triple_sel,
                              exp_sel, tar_sel, ptm_model_name, llm_type, llm_temp)
        compute_llm_output_distribution(
            dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
            ptm_model_name, llm_type, llm_temp, max_rep, lower_bound, upper_bound, train_percentage)
        print('Overall process status: (3/3)')
        eval_mode = 'rep'
        evaluate(dataset_name, eval_mode)
    else:
        old_ptm_model_name = get_ptm_model_name_with_round(ptm_model_name, round - 1)
        new_ptm_model_name = get_ptm_model_name_with_round(ptm_model_name, round)
        ## generate distribution data
        if not tar_sel.endswith('ignore'):
            print('Overall process status: (1/6)')
            generate_training_data_from_distribution(dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel,
                                                     tar_sel, old_ptm_model_name, llm_type, llm_temp,
                                                     max_rep, lower_bound, upper_bound, train_percentage)
            ## generate training and dev data
            raw_data_path, train_data_path, dev_data_path = get_raw_train_dev_data_path(
                dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
                old_ptm_model_name, llm_type, llm_temp, max_rep, lower_bound, upper_bound, train_percentage)
            seed = 42
            ratio = 0.8
            print('Overall process status: (2/6)')
            split_data(train_data_path, dev_data_path, raw_data_path, seed, ratio)
            ## train new ptm_model based on the traning and dev data
            print('Overall process status: (3/6)')
            train_ptm_model(train_data_path, dev_data_path, old_ptm_model_name, new_ptm_model_name, dataset_name,
                            llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
                            llm_type, llm_temp, max_rep, lower_bound, train_bound)
        ## LLM annotation and evaluation
        if debug_mode:
            # debugging training module
            upper_bound = 100
        print('Overall process status: (4/6)')
        ea_by_llm_recursively(
            dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
            new_ptm_model_name, llm_type, llm_model, llm_temp, max_rep, lower_bound, upper_bound, train_percentage
        )
        print('Overall process status: (5/6)')
        move_to_processed_dir(dataset_name, llm_mode, ea_data_mode, triple_sel,
                              exp_sel, tar_sel, new_ptm_model_name, llm_type, llm_temp)
        compute_llm_output_distribution(
            dataset_name, llm_mode, ea_data_mode, triple_sel, exp_sel, tar_sel,
            new_ptm_model_name, llm_type, llm_temp, max_rep, lower_bound, upper_bound, train_percentage)
        print('Overall process status: (6/6)')
        eval_mode = 'rep'
        evaluate(dataset_name, eval_mode)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform HLMEA')
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
                        help='dze, dje, dfe, dw, dy, ddev, dfev, ddev100k, dfev100k')
    parser.add_argument('-p', '--ptm_model_name', type=str, required=True, help='e5, labse, mpnet, minilm')
    parser.add_argument('-l', '--llm_type', type=str, required=True, help='gpt3.5, gpt4, qwen')
    parser.add_argument('-m', '--ea_data_mode', type=str, required=True, help='train, test')
    parser.add_argument('-r', '--round', type=int, required=True, help='0 to k')
    parser.add_argument('--triple_sel', type=str, default='ptm-5-triple-freq',
                        help='ptm-5-triple-freq, ptm-5-desc-sparql')
    parser.add_argument('--tar_sel', type=str, default='freq-5',
                        help='freq-k, rand-k')
    parser.add_argument('--debug_mode', type=bool, default=False, help='True or False')
    parser.add_argument('--llm_mode', type=str, default='icl', help='icl or zs')
    parser.add_argument('--max_rep', type=int, default=5)
    parser.add_argument('--exp_sel', type=str, default='')
    parser.add_argument('--llm_temp', type=str, default='0')
    parser.add_argument('--train_percentage', type=float, default=0.9)
    args = parser.parse_args()

    main(args.dataset_name, args.ptm_model_name, args.llm_type, args.ea_data_mode, args.round,
         args.triple_sel, args.tar_sel, args.debug_mode, args.llm_mode, args.max_rep, args.exp_sel,
         args.llm_temp, args.train_percentage)

    # main(dataset_name='dze', ptm_model_name='labse', llm_type='gpt3.5', ea_data_mode='train', round=0,
    #      tar_sel='freq-3', triple_sel='ptm-5-triple-freq', debug_mode=False, llm_mode='zs', max_rep=5, exp_sel='',
    #      llm_temp='0', train_percentage=0.9)
