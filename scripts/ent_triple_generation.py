import json
import os

from tqdm import tqdm

from python_scripts.llm_classification import get_attr_and_rel_filenames, get_random_samples_from_dict
from utility import read_groundtruth_with_mode, get_element_triples, get_frequent_rel_and_attr, \
    read_ent_first_attr_triples, read_ent_first_rel_triples
import argparse


def generate_freq_attr_rel_triple(ent_list: list, attr_triples: dict, rel_out_triples: dict, rel_in_triples: dict,
                                  ent_sorted_attr_list: list, ent_sorted_rel_out_list: list,
                                  ent_sorted_rel_in_list: list, num_triple: int):
    ent_triple_dict = {}
    print('Generating frequent ent triples ... ')
    for ent in tqdm(ent_list):
        ## add attribute triples
        triple_direction = 'out'
        element_type = 'attr'
        ent_triple_dict = add_single_element_triple(
            ent_triple_dict, ent, attr_triples, ent_sorted_attr_list, element_type, triple_direction, num_triple)
        ## add out relation triples
        element_type = 'rel'
        ent_triple_dict = add_single_element_triple(
            ent_triple_dict, ent, rel_out_triples, ent_sorted_rel_out_list,
            element_type, triple_direction, num_triple)
        ## add in relation triples
        triple_direction = 'in'
        ent_triple_dict = add_single_element_triple(
            ent_triple_dict, ent, rel_in_triples, ent_sorted_rel_in_list,
            element_type, triple_direction, num_triple)

    return ent_triple_dict


def generate_rand_attr_rel_triple(tar_ent_list: list, num_triple: int,
                                  attr_dict: dict, rel_out_dict: dict, rel_in_dict: dict):
    ent_triple_dict = {}
    print('Generating random triples ... ')
    for tar_ent in tqdm(tar_ent_list):
        ent_name = tar_ent.split("/")[-1]
        # attr_filename, rel_filename = get_attr_and_rel_filenames(dataset_name, tar_ent_list)
        # attr_dict = read_ent_first_attr_triples(dataset_name, attr_filename)
        if tar_ent in attr_dict.keys():
            ent_attr_dict = attr_dict[tar_ent]
            if num_triple < len(ent_attr_dict):
                ent_attr_dict = get_random_samples_from_dict(ent_attr_dict, num_triple)
            for attr in ent_attr_dict:
                attr_name = attr.split("/")[-1]
                value = ent_attr_dict[attr][0]
                ent_triple_text = '(%s, %s, %s)' % (ent_name, attr_name, value)
                if tar_ent not in ent_triple_dict:
                    ent_triple_dict[tar_ent] = [ent_triple_text]
                else:
                    ent_triple_dict[tar_ent].append(ent_triple_text)
        # rel_out_dict, rel_in_dict = read_ent_first_rel_triples(dataset_name, rel_filename)
        if tar_ent in rel_out_dict.keys():
            ent_rel_out_dict = rel_out_dict[tar_ent]
            if num_triple < len(ent_rel_out_dict):
                ent_rel_out_dict = get_random_samples_from_dict(ent_rel_out_dict, num_triple)
            for rel_out in ent_rel_out_dict:
                rel_out_name = rel_out.split("/")[-1]
                object = ent_rel_out_dict[rel_out][0]
                object_name = object.split("/")[-1]
                ent_triple_text = '(%s, %s, %s)' % (tar_ent, rel_out_name, object_name)
                if tar_ent not in ent_triple_dict:
                    ent_triple_dict[tar_ent] = [ent_triple_text]
                else:
                    ent_triple_dict[tar_ent].append(ent_triple_text)
        if tar_ent in rel_in_dict.keys():
            ent_rel_in_dict = rel_in_dict[tar_ent]
            if num_triple < len(ent_rel_in_dict):
                ent_rel_in_dict = get_random_samples_from_dict(ent_rel_in_dict, num_triple)
            for rel_in in ent_rel_in_dict:
                rel_in_name = rel_in.split("/")[-1]
                suject = ent_rel_in_dict[rel_in][0]
                suject_name = suject.split("/")[-1]
                ent_triple_text = '(%s, %s, %s)' % (suject_name, rel_in_name, tar_ent)
                if tar_ent not in ent_triple_dict:
                    ent_triple_dict[tar_ent] = [ent_triple_text]
                else:
                    ent_triple_dict[tar_ent].append(ent_triple_text)

    return ent_triple_dict


def add_single_element_triple(ent_triple_dict: dict, ent: str, element_triples: dict,
                              ent_sorted_element_list: list, element_type: str, triple_direction: str,
                              num_triple: int):
    if ent in element_triples:
        ent_element_dict = element_triples[ent]
        triple_index = 0
        for element_index in range(len(ent_sorted_element_list)):
            cur_element = ent_sorted_element_list[element_index]
            if cur_element in ent_element_dict:
                if element_type == 'attr':
                    so = ent_element_dict[cur_element][0]
                elif element_type == 'rel':
                    so = ent_element_dict[cur_element][0].split("/")[-1]
                else:
                    raise RuntimeError('Unknown element_type: %s' % element_type)

                ent_name = ent.split("/")[-1]
                element_name = cur_element.split("/")[-1]
                if triple_direction == 'out':
                    triple = '(%s, %s, %s)' % (ent_name, element_name, so)
                elif triple_direction == 'in':
                    triple = '(%s, %s, %s)' % (so, element_name, ent_name)
                else:
                    raise RuntimeError('Unknown triple_direction: %s' % triple_direction)

                if ent not in ent_triple_dict:
                    ent_triple_dict[ent] = [triple]
                else:
                    ent_triple_dict[ent].append(triple)
                triple_index += 1
                if triple_index >= num_triple:
                    break

    return ent_triple_dict


def generate_ent_triple(dataset_name: str, ea_data_mode: str, num_triple: int, triple_strategy: str):
    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    ent1_list = list(gt_dict.keys())
    ent2_list = list(gt_dict.values())

    attr_triples1, rel_out_triples1, rel_in_triples1, attr_triples2, rel_out_triples2, rel_in_triples2 = \
        get_element_triples(dataset_name)

    if triple_strategy == 'freq':
        ent1_sorted_attr_list, ent1_sorted_rel_out_list, ent1_sorted_rel_in_list, \
            ent2_sorted_attr_list, ent2_sorted_rel_out_list, ent2_sorted_rel_in_list = \
            get_frequent_rel_and_attr(dataset_name, ea_data_mode)

        ent1_triple_dict = generate_freq_attr_rel_triple(
            ent1_list, attr_triples1, rel_out_triples1, rel_in_triples1,
            ent1_sorted_attr_list, ent1_sorted_rel_out_list, ent1_sorted_rel_in_list, num_triple)
        ent2_triple_dict = generate_freq_attr_rel_triple(
            ent2_list, attr_triples2, rel_out_triples2, rel_in_triples2,
            ent2_sorted_attr_list, ent2_sorted_rel_out_list, ent2_sorted_rel_in_list, num_triple)
    elif triple_strategy == 'rand':
        ent1_triple_dict = generate_rand_attr_rel_triple(ent1_list, num_triple,
                                                         attr_triples1, rel_out_triples1, rel_in_triples1)
        ent2_triple_dict = generate_rand_attr_rel_triple(ent2_list, num_triple,
                                                         attr_triples2, rel_out_triples2, rel_in_triples2)
    else:
        raise RuntimeError('Unknown tar_strategy: %s' % triple_strategy)

    output_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_triple_dict', dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ent1_triple_file_name = 'ent1_triple=%s=%d=%s.json' % (ea_data_mode, num_triple, triple_strategy)
    ent1_triple_file_path = os.path.join(output_dir, ent1_triple_file_name)
    with open(ent1_triple_file_path, 'w', encoding='utf-8') as f:
        json.dump(ent1_triple_dict, f, ensure_ascii=False, indent=4)
        print('Generated file ... %s' % ent1_triple_file_path)

    ent2_triple_file_name = 'ent2_triple=%s=%d=%s.json' % (ea_data_mode, num_triple, triple_strategy)
    ent2_triple_file_path = os.path.join(output_dir, ent2_triple_file_name)
    with open(ent2_triple_file_path, 'w', encoding='utf-8') as f:
        json.dump(ent2_triple_dict, f, ensure_ascii=False, indent=4)
        print('Generated file ... %s' % ent2_triple_file_path)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TRE triples')
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
                        help='dze, dje, dfe, dw, dy, ddev, dfev, ddev100k, dfev100k')
    parser.add_argument('--ea_data_mode', type=str, default='all', help='all, train, test')
    parser.add_argument('--num_triple', type=int, default=5, help='5, 3, 7')
    parser.add_argument('--triple_strategy', type=str, default='freq', help='freq, rand')
    args = parser.parse_args()

    generate_ent_triple(args.dataset_name, args.ea_data_mode, args.num_triple, args.triple_strategy)
