import json
import os
import time

from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from tqdm import tqdm

from utility import read_groundtruth_with_mode


def get_entity_image_url(entity_iri: str, language: str):
    os.environ["http_proxy"] = "http://10.105.20.64:7890"
    os.environ["https_proxy"] = "http://10.105.20.64:7890"

    if language == 'en':
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    elif language == 'fr':
        sparql = SPARQLWrapper("http://fr.dbpedia.org/sparql")
    else:
        raise RuntimeError("Unknown language: %s" % language)

    query = """
    SELECT ?image
    WHERE {
      <%s> <http://dbpedia.org/ontology/thumbnail> ?image .
    }
    LIMIT 1
    """ % entity_iri
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if results["results"]["bindings"]:
        image_url = results["results"]["bindings"][0]["image"]["value"]
        return image_url
    else:
        return None


def download_image(image_url: str, output_dir: str, index: int):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(image_url, headers=headers)

    if response.status_code == 200:
        img_file_path = os.path.join(output_dir, '%d.png' % index)
        with open(img_file_path, 'wb') as file:
            file.write(response.content)
        print('Image downloaded successfully: %s' % img_file_path)
    else:
        failed_file_path = os.path.join(output_dir, '%d_download_failure.txt' % index)
        with open(failed_file_path, 'w') as file:
            pass
        print('(%d) Failed to download image. HTTP Status Code: %d' % (index, response.status_code))

    return 0


def generate_ent_img_index(output_dir: str, tar_ent_list_name: str, ent_list: list):
    ent_img_index_path = os.path.join(output_dir, tar_ent_list_name + '_img_index.json')
    if os.path.exists(ent_img_index_path):
        return 0
    else:
        ent_img_index_dict = {}
        for index in range(len(ent_list)):
            ent = ent_list[index]
            ent_img_index_dict[ent] = index
        with open(ent_img_index_path, 'w', encoding='utf-8') as f:
            json.dump(ent_img_index_dict, f, ensure_ascii=False, indent=4)

    return 0


def retrieve_ent_img(dataset_name: str, ea_data_mode: str, tar_ent_list_name: str, lower_bound: int, upper_bound: int):
    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    if tar_ent_list_name == 'ent1':
        ent_list = list(gt_dict.keys())
        language = 'en'
    elif tar_ent_list_name == 'ent2':
        ent_list = list(gt_dict.values())
        language = 'fr'
    else:
        raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)

    output_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_raw_image', dataset_name, tar_ent_list_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_ent_img_index(output_dir, tar_ent_list_name, ent_list)

    for index in tqdm(range(lower_bound, upper_bound), desc=" %s_image from %d to %d" %
                                                            (tar_ent_list_name, lower_bound, upper_bound)):
        time.sleep(0.01)
        ent = ent_list[index]
        files = os.listdir(output_dir)
        flag = False
        for file in files:
            if file.split('_')[0] == str(index):
                flag = True
                break

        if not flag:
            try:
                image_url = get_entity_image_url(ent, language)
                if image_url:
                    time.sleep(1)
                    download_image(image_url, output_dir, index)
                else:
                    file_name = '%d_no_url.txt' % index
                    file_path = os.path.join(output_dir, file_name)
                    with open(file_path, 'w') as file:
                        pass
                    print("(%d) No image url returned for the entity: %s" % (index, ent))
            except Exception as e:
                # Catch any exceptions and print the error message
                print("An error occurred:", e)
                print('while processing index: %d and entity: %s' % (index, ent))
                time.sleep(10)
                retrieve_ent_img(dataset_name, ea_data_mode, tar_ent_list_name, index, upper_bound)

    return 0


if __name__ == "__main__":
    dataset_name = 'DBP100K_FR_EN_V1'
    ea_data_mode = 'all'
    tar_ent_list_name = 'ent1'
    lower_bound = 0
    upper_bound = 100000
    retrieve_ent_img(dataset_name, ea_data_mode, tar_ent_list_name, lower_bound, upper_bound)
