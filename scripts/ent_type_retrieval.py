import json
import os
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utility import read_groundtruth_with_mode


def extract_resouce(soup):
    target_class_list = ['page-resource-uri', 'text-muted']
    for target_class in target_class_list:
        resource_uri_div = soup.find('div', class_=target_class)
        if not isinstance(resource_uri_div, type(None)):
            return resource_uri_div

    return None


def get_dbpedia_type(ent_iri: str):
    os.environ["http_proxy"] = "http://10.105.20.64:7890"
    os.environ["https_proxy"] = "http://10.105.20.64:7890"

    try:
        response = requests.get(ent_iri)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find the div with class
            resource_uri_div = extract_resouce(soup)
            if resource_uri_div:
                type_hyperlink = resource_uri_div.find('a')['href'] if resource_uri_div.find('a') else None
                if type_hyperlink.startswith('http'):
                    ent_type = type_hyperlink.split('/')[-1]
                else:
                    full_text = resource_uri_div.get_text(strip=True)
                    # Extract the type part from the text
                    type_str_start = full_text.find("An Entity of Type :") + len("An Entity of Type :")
                    type_str_end = full_text.find(",", type_str_start)
                    ent_type = full_text[type_str_start:type_str_end].strip()

                return ent_type
            else:
                raise RuntimeError("Div with class not found for ent: %s" % ent_iri)
        elif response.status_code == 404:
            print('HTTP Status Code 404 for ent: %s' % ent_iri)
            return '404'
        else:
            raise RuntimeError(f"Error fetching page: HTTP Status Code {response.status_code} for ent: %s" % ent_iri)
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e} for ent: %s" % ent_iri)


def retrieve_ent_type(dataset_name: str, ea_data_mode: str, tar_ent_list_name: str, lower_bound: int, upper_bound: int):
    ent_type_dict = {}

    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    if tar_ent_list_name == 'ent1':
        ent_list = list(gt_dict.keys())
    elif tar_ent_list_name == 'ent2':
        ent_list = list(gt_dict.values())
    else:
        raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)

    index = lower_bound
    for ent in tqdm(ent_list[lower_bound:upper_bound], desc=" ent_type from %d to %d" % (lower_bound, upper_bound)):
        if lower_bound <= index < upper_bound:
            time.sleep(0.01)
            try:
                ent_type = get_dbpedia_type(ent)
                ent_type_dict[ent] = ent_type
                index += 1
            except:
                print('Type retrieval failed (%d/%d): %s' % (index, upper_bound, ent))
                break

    if index > lower_bound:
        ent_type_dict_file_name = ('ent_type_dict_by_retrieval=%s=%s=%s=%d=%d.json' %
                                   (dataset_name, ea_data_mode, tar_ent_list_name, lower_bound, index))
        ent_type_dict_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_type_dict', dataset_name)
        if not os.path.exists(ent_type_dict_dir):
            os.makedirs(ent_type_dict_dir)
        ent_type_dict_path = os.path.join(ent_type_dict_dir, ent_type_dict_file_name)
        with open(ent_type_dict_path, 'w', encoding='utf-8') as f:
            json.dump(ent_type_dict, f, ensure_ascii=False, indent=4)
        print('Output generated successfully: %s' % ent_type_dict_path)

    return 0


def get_current_upper_bound(dataset_name: str, ea_data_mode: str, tar_ent_list_name: str):
    upper_bound = 0
    model_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_type_dict', dataset_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        return 0
    filenames = os.listdir(model_output_dir)
    if len(filenames) == 0:
        return 0
    else:
        prefix = ('ent_type_dict_by_retrieval=%s=%s=%s' % (dataset_name, ea_data_mode, tar_ent_list_name))
        for file_name in filenames:
            suffix = os.path.splitext(file_name)[-1]
            if file_name.startswith(prefix) and suffix == '.json':
                base_name = os.path.splitext(file_name)[0]
                split_list = base_name.split('=')
                cur_upper = int(split_list[-1])
                if upper_bound < cur_upper:
                    upper_bound = cur_upper

    return upper_bound


def retrieve_ent_type_recursively(dataset_name: str, ea_data_mode: str, tar_ent_list_name: str,
                                  lower_bound: int, upper_bound: int):
    cur_upper = get_current_upper_bound(dataset_name, ea_data_mode, tar_ent_list_name)
    while cur_upper < upper_bound:
        if cur_upper > lower_bound:
            lower_bound = cur_upper

        print('Start to process ent_type_dict_by_retrieval=%s=%s=%s=%d=%d.json' %
              (dataset_name, ea_data_mode, tar_ent_list_name, lower_bound, upper_bound))
        retrieve_ent_type(dataset_name, ea_data_mode, tar_ent_list_name, lower_bound, upper_bound)

        cur_upper = get_current_upper_bound(dataset_name, ea_data_mode, tar_ent_list_name)
        time.sleep(1)

    return 0


def run_in_parallel(dataset_name, ea_data_mode, tar_ent_list_name, lower_bound, upper_bound, k):
    # Calculate the step size for each thread based on k
    step = (upper_bound - lower_bound) // k
    futures = []

    with ThreadPoolExecutor(max_workers=k) as executor:
        for i in range(k):
            # Calculate the specific range for each thread
            start = lower_bound + i * step
            end = start + step if i < k - 1 else upper_bound
            # Submit the task to the thread pool
            futures.append(
                executor.submit(
                    retrieve_ent_type_recursively, dataset_name, ea_data_mode, tar_ent_list_name, start, end)
            )

        # Optionally, wait for all submitted tasks to complete and handle their results
        for future in as_completed(futures):
            result = future.result()
            print(f"Task completed with result: {result}")


if __name__ == '__main__':
    # ent_iri = 'http://dbpedia.org/resource/Illumination_(Tristania_album)'
    # ent_type = get_dbpedia_type(ent_iri)
    # print(ent_type)

    ## DBP15K_DE_EN_V1, DW15K_V1, DBP15K_FR_EN, DBP15K_JA_EN, DBP15K_ZH_EN
    dataset_name_list = ['DBP100K_FR_EN_V1']
    tar_ent_list_names = ['ent1']
    ea_data_mode = 'all'
    lower_bound = 0
    upper_bound = 100000
    k = 100

    for dataset_name in dataset_name_list:
        for tar_ent_list_name in tar_ent_list_names:
            retrieve_ent_type_recursively(dataset_name, ea_data_mode, tar_ent_list_name, lower_bound, upper_bound)
