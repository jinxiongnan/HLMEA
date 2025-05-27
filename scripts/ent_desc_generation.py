import json
import os

from tqdm import tqdm

from python_scripts.utility import read_groundtruth_with_mode, get_element_triples


# def generate_ent1_desc_by_llm(dataset_name: str, ea_data_mode: str, triple_sel: str,
#                               llm_type: str, llm_model: str, llm_temp: str, lower_bound: int, upper_bound: int):
#     gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
#     ent1_list = list(gt_dict.keys())
#
#     ent1_desc_dict = {}
#
#     index = lower_bound
#     for ent1 in tqdm(ent1_list[lower_bound:upper_bound], desc='ent1_desc'):
#         prompt = prompt_generation(triple_sel, ent1, 'ent1')
#         try:
#             desc = get_response_from_llm(llm_type, prompt, llm_model, llm_temp)
#             ent1_desc_dict[ent1] = desc
#             index += 1
#         except:
#             break
#
#     if index > lower_bound:
#         if llm_type in ['baichuan2-v100', 'baichuan2-a6000']:
#             llm_type_alias = 'baichuan2'
#         else:
#             llm_type_alias = llm_type
#         ent1_desc_dict_file_name = 'ent_desc_dict_by_llm=%s=%s=ent1=%d=%d.json' % \
#                                    (ea_data_mode, llm_type_alias, lower_bound, upper_bound)
#         ent_desc_dict_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name)
#         if not os.path.exists(ent_desc_dict_dir):
#             os.makedirs(ent_desc_dict_dir)
#         ent1_desc_dict_path = os.path.join(ent_desc_dict_dir, ent1_desc_dict_file_name)
#         with open(ent1_desc_dict_path, 'w', encoding='utf-8') as f:
#             json.dump(ent1_desc_dict, f, ensure_ascii=False, indent=4)
#         print('Output generated successfully: %s' % ent1_desc_dict_path)
#
#     return 0
#
#
# def generate_ent2_desc_by_llm(dataset_name: str, ea_data_mode: str, triple_sel: str,
#                               llm_type: str, llm_model: str, llm_temp: str, lower_bound: int, upper_bound: int):
#     gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
#     ent2_list = list(gt_dict.values())
#
#     ent2_desc_dict = {}
#
#     index = lower_bound
#     for ent2 in tqdm(ent2_list[lower_bound:upper_bound], desc='ent2_desc'):
#         prompt = prompt_generation(triple_sel, ent2, 'ent2')
#         try:
#             desc = get_response_from_llm(llm_type, prompt, llm_model, llm_temp)
#             ent2_desc_dict[ent2] = desc
#             index += 1
#         except:
#             break
#
#     if index > lower_bound:
#         if llm_type in ['baichuan2-v100', 'baichuan2-a6000']:
#             llm_type_alias = 'baichuan2'
#         else:
#             llm_type_alias = llm_type
#         ent2_desc_dict_file_name = 'ent_desc_dict_by_llm=%s=%s=ent2=%d=%d.json' % \
#                                    (ea_data_mode, llm_type_alias, lower_bound, upper_bound)
#         ent_desc_dict_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name)
#         if not os.path.exists(ent_desc_dict_dir):
#             os.makedirs(ent_desc_dict_dir)
#         ent2_desc_dict_path = os.path.join(ent_desc_dict_dir, ent2_desc_dict_file_name)
#         with open(ent2_desc_dict_path, 'w', encoding='utf-8') as f:
#             json.dump(ent2_desc_dict, f, ensure_ascii=False, indent=4)
#         print('Output generated successfully: %s' % ent2_desc_dict_path)
#
#     return 0
#
#
# def get_current_upper_bound(dataset_name: str, ea_data_mode: str, llm_type: str, tar_ent_list_name: str):
#     upper_bound = 0
#     model_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name)
#     if not os.path.exists(model_output_dir):
#         os.makedirs(model_output_dir)
#         return 0
#     filenames = os.listdir(model_output_dir)
#     if len(filenames) == 0:
#         return 0
#     else:
#         if llm_type in ['baichuan2-v100', 'baichuan2-a6000']:
#             llm_type_alias = 'baichuan2'
#         else:
#             llm_type_alias = llm_type
#         prefix = ('ent_desc_dict_by_llm=%s=%s=%s' % (ea_data_mode, llm_type_alias, tar_ent_list_name))
#         for file_name in filenames:
#             suffix = os.path.splitext(file_name)[-1]
#             if file_name.startswith(prefix) and suffix == '.json':
#                 base_name = os.path.splitext(file_name)[0]
#                 split_list = base_name.split('=')
#                 cur_upper = int(split_list[-1])
#                 if upper_bound < cur_upper:
#                     upper_bound = cur_upper
#
#     return upper_bound
#
#
# def generate_ent_desc_by_llm_recursively(dataset_name: str, ea_data_mode: str, triple_sel: str,
#                                          llm_type: str, llm_model: str, llm_temp: str,
#                                          lower_bound: int, upper_bound: int):
#     tar_ent_list_name = 'ent1'
#     cur_upper = get_current_upper_bound(dataset_name, ea_data_mode, llm_type, tar_ent_list_name)
#     while cur_upper < upper_bound:
#         if cur_upper > lower_bound:
#             lower_bound = cur_upper
#
#         print('Start to process ent1_desc_dict_by_llm=%s=%s=%d=%d.json' %
#               (dataset_name, llm_type, lower_bound, upper_bound))
#         generate_ent1_desc_by_llm(dataset_name, ea_data_mode, triple_sel, llm_type, llm_model, llm_temp,
#                                   lower_bound, upper_bound)
#
#         cur_upper = get_current_upper_bound(dataset_name, ea_data_mode, llm_type, tar_ent_list_name)
#         time.sleep(1)
#
#     tar_ent_list_name = 'ent2'
#     cur_upper = get_current_upper_bound(dataset_name, ea_data_mode, llm_type, tar_ent_list_name)
#     while cur_upper < upper_bound:
#         if cur_upper > lower_bound:
#             lower_bound = cur_upper
#
#         print('Start to process ent2_desc_dict_by_llm=%s=%s=%d=%d.json' %
#               (dataset_name, llm_type, lower_bound, upper_bound))
#         generate_ent2_desc_by_llm(dataset_name, ea_data_mode, triple_sel, llm_type, llm_model, llm_temp,
#                                   lower_bound, upper_bound)
#
#         cur_upper = get_current_upper_bound(dataset_name, ea_data_mode, llm_type, tar_ent_list_name)
#         time.sleep(1)
#
#     return 0
#
#
# def complete_ent_desc_by_llm(dataset_name: str, ea_data_mode: str, triple_sel: str, target_ent_list: str,
#                              llm_type: str, llm_model: str, llm_temp: str):
#     if llm_type in ['baichuan2-v100', 'baichuan2-a6000']:
#         llm_type_alias = 'baichuan2'
#     else:
#         llm_type_alias = llm_type
#     ent_desc_file_path = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name,
#                                       'ent_desc_dict_by_llm=%s=%s=%s.json' %
#                                       (ea_data_mode, llm_type_alias, target_ent_list))
#
#     print('statistics BEFORE completion are:')
#     analyze_ent_desc_dict(ent_desc_file_path)
#
#     with open(ent_desc_file_path, 'r', encoding='utf-8') as f:
#         ent_desc_dict = json.load(f)
#
#     print('Start to complete %s' % ent_desc_file_path)
#     num_not_valid = 0
#     for ent in tqdm(ent_desc_dict):
#         desc = ent_desc_dict[ent]
#         if not is_desc_valid(desc):
#             num_not_valid += 1
#             prompt = prompt_generation(triple_sel, ent, target_ent_list)
#             try:
#                 print('(%d) generating desc for ent "%s" using llm "%s"' % (num_not_valid, ent, llm_type))
#                 desc = get_response_from_llm(llm_type, prompt, llm_model, llm_temp)
#             except:
#                 desc = 'error'
#             if not is_desc_valid(desc):
#                 print('(%d) Desc "%s" is STILL not valid for ent: %s' % (num_not_valid, desc, ent))
#             else:
#                 ent_desc_dict[ent] = desc
#
#     with open(ent_desc_file_path, 'w', encoding='utf-8') as f:
#         json.dump(ent_desc_dict, f, ensure_ascii=False, indent=4)
#     print('Output generated successfully: %s' % ent_desc_file_path)
#
#     print('statistics AFTER completion are:')
#     analyze_ent_desc_dict(ent_desc_file_path)
#
#     return 0


def add_single_element_triple(ent_desc_dict: dict, ent: str,
                              element_triples: dict, element_type: str, triple_direction: str):
    if ent in element_triples:
        ent_element_dict = element_triples[ent]
        for ent_element in ent_element_dict:
            so_list = ent_element_dict[ent_element]
                ## so = ent_element_dict[cur_element][0].split("/")[-1]

            ent_name = ent.split("/")[-1]
            element_name = ent_element.split("/")[-1]
            for so in so_list:
                if element_type == 'rel':
                    so = so.split("/")[-1]
                ## generate triple
                if triple_direction == 'out':
                    triple = '(%s, %s, %s)' % (ent_name, element_name, so)
                elif triple_direction == 'in':
                    triple = '(%s, %s, %s)' % (so, element_name, ent_name)
                else:
                    raise RuntimeError('Unknown triple_direction: %s' % triple_direction)
                ## add triple to ent_desc_dict
                if ent not in ent_desc_dict:
                    ent_desc_dict[ent] = [triple]
                else:
                    ent_desc_dict[ent].append(triple)

    return ent_desc_dict


def generate_attr_rel_triple(ent_list: list, attr_triples: dict, rel_out_triples: dict, rel_in_triples: dict):
    ent_desc_dict = {}
    for ent in tqdm(ent_list):
        ## add attribute triples
        triple_direction = 'out'
        element_type = 'attr'
        ent_desc_dict = add_single_element_triple(ent_desc_dict, ent, attr_triples, element_type, triple_direction)
        ## add out relation triples
        element_type = 'rel'
        ent_desc_dict = add_single_element_triple(ent_desc_dict, ent, rel_out_triples, element_type, triple_direction)
        ## add in relation triples
        triple_direction = 'in'
        ent_desc_dict = add_single_element_triple(ent_desc_dict, ent, rel_in_triples, element_type, triple_direction)

    return ent_desc_dict


def generate_ent_desc_based_on_triple(dataset_name: str, ea_data_mode: str):
    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    ent1_list = list(gt_dict.keys())
    ent2_list = list(gt_dict.values())

    attr_triples1, rel_out_triples1, rel_in_triples1, attr_triples2, rel_out_triples2, rel_in_triples2 = \
        get_element_triples(dataset_name)

    print('Generating entity 1 descriptions based on triples ... ')
    ent1_desc_dict = generate_attr_rel_triple(ent1_list, attr_triples1, rel_out_triples1, rel_in_triples1)
    print('Generating entity 2 descriptions based on triples ... ')
    ent2_desc_dict = generate_attr_rel_triple(ent2_list, attr_triples2, rel_out_triples2, rel_in_triples2)

    output_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ent1_desc_file_name = 'ent1_desc_based_on_triple=%s.json' % ea_data_mode
    ent1_desc_file_path = os.path.join(output_dir, ent1_desc_file_name)
    with open(ent1_desc_file_path, 'w', encoding='utf-8') as f:
        json.dump(ent1_desc_dict, f, ensure_ascii=False, indent=4)
        print('Generated file ... %s' % ent1_desc_file_path)

    ent2_desc_file_name = 'ent2_desc_based_on_triple=%s.json' % ea_data_mode
    ent2_desc_file_path = os.path.join(output_dir, ent2_desc_file_name)
    with open(ent2_desc_file_path, 'w', encoding='utf-8') as f:
        json.dump(ent2_desc_dict, f, ensure_ascii=False, indent=4)
        print('Generated file ... %s' % ent2_desc_file_path)

    return 0


if __name__ == '__main__':
    ## ['DW15K_V1', 'DBP15K_DE_EN_V1', 'DBP15K_FR_EN', 'DBP15K_JA_EN', 'DBP15K_ZH_EN', 'DBP100K_FR_EN_V1']
    dataset_name_list = ['DBP100K_FR_EN_V1']
    ea_data_mode = 'all'
    # tar_ent_list_name = 'ent1'
    # triple_sel = 'freq-10'
    # llm_type = 'baichuan2-a6000'
    # llm_model = 'gpt-3.5-turbo-1106'
    # llm_temp = '0'
    # lower_bound = 0
    # upper_bound = 3000

    # generate_ent_desc_by_llm_recursively(dataset_name, ea_data_mode, triple_sel, llm_type, llm_model, llm_temp,
    #                                      lower_bound, upper_bound)
    # complete_ent_desc_by_llm(dataset_name, ea_data_mode, tar_ent_list_name, llm_type, llm_model, llm_temp)

    for dataset_name in dataset_name_list:
        generate_ent_desc_based_on_triple(dataset_name, ea_data_mode)
