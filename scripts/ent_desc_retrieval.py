import json
import os
import time
import requests

from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

from utility import read_groundtruth_with_mode, is_desc_valid, analyze_ent_desc_dict, \
    get_kg_and_entLang


def call_sparql_dbpedia(ent_iri: str, language: str):
    os.environ["http_proxy"] = "http://10.105.20.64:7890"
    os.environ["https_proxy"] = "http://10.105.20.64:7890"

    # Initialize the SPARQL wrapper and set the endpoint
    if language == 'en':
        endpoint = "http://dbpedia.org/sparql"
    elif language == 'de':
        endpoint = "http://de.dbpedia.org/sparql"
    elif language == 'fr':
        endpoint = "http://fr.dbpedia.org/sparql"
    elif language == 'ja':
        endpoint = "http://ja.dbpedia.org/sparql"
    elif language == 'zh':
        endpoint = "http://zh.dbpedia.org/sparql"
    else:
        raise RuntimeError('Unknown dbpedia language %s' % language)
    sparql = SPARQLWrapper(endpoint)

    # SPARQL query to get the description in German
    query = """
    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?description
    WHERE {
        <%s> dbo:abstract ?description .
        FILTER (lang(?description) = '%s')
    }
    LIMIT 1
    """ % (ent_iri, language)

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query and extract the result
    try:
        results = sparql.query().convert()
    except:
        return "error"

    if results["results"]["bindings"]:
        description = results["results"]["bindings"][0]["description"]["value"]
        return description
    else:
        return None


def call_sparql_wikidata(ent_iri: str):
    qid = ent_iri.split('/')[-1]

    # SPARQL query
    query = """
    SELECT ?entityDescription WHERE {
      BIND(wd:%s AS ?entity)
      OPTIONAL { ?entity schema:description ?entityDescription . FILTER(LANG(?entityDescription) = "en") }
    }
    """ % qid

    # URL for the Wikidata SPARQL endpoint
    url = "https://query.wikidata.org/sparql"

    os.environ["http_proxy"] = "http://10.105.20.64:7890"
    os.environ["https_proxy"] = "http://10.105.20.64:7890"

    # Send the request
    try:
        response = requests.get(url, params={'query': query, 'format': 'json'})
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            descriptions = data['results']['bindings']
            if descriptions:
                desc = descriptions[0]['entityDescription']['value']
                return desc
            else:
                return None
    except:
        return 'error'


def retrieve_ent_desc(dataset_name: str, ea_data_mode: str, tar_ent_list_name: str, lower_bound: int, upper_bound: int):
    ent_desc_dict = {}

    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    if tar_ent_list_name == 'ent1':
        ent_list = list(gt_dict.keys())
    elif tar_ent_list_name == 'ent2':
        ent_list = list(gt_dict.values())
    else:
        raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)

    kg, ent_lang = get_kg_and_entLang(dataset_name, tar_ent_list_name)

    index = lower_bound
    for ent in tqdm(ent_list[lower_bound:upper_bound], desc=" ent_desc"):
        if lower_bound <= index < upper_bound:
            time.sleep(0.01)
            try:
                if kg == 'dbpedia':
                    desc = call_sparql_dbpedia(ent, ent_lang)
                elif kg == 'wikidata':
                    desc = call_sparql_wikidata(ent)
                else:
                    raise RuntimeError('Unknown kg: %s' % kg)
                ent_desc_dict[ent] = desc
                index += 1
            except:
                print('Description retrieval failed (%d/%d): %s' % (index, upper_bound, ent))
                break

    if index > lower_bound:
        ent_desc_dict_file_name = ('ent_desc_dict_by_retrieval=%s=%s=%d=%d.json' %
                                   (ea_data_mode, tar_ent_list_name, lower_bound, index))
        ent_desc_dict_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name)
        if not os.path.exists(ent_desc_dict_dir):
            os.makedirs(ent_desc_dict_dir)
        ent_desc_dict_path = os.path.join(ent_desc_dict_dir, ent_desc_dict_file_name)
        with open(ent_desc_dict_path, 'w', encoding='utf-8') as f:
            json.dump(ent_desc_dict, f, ensure_ascii=False, indent=4)
        print('Output generated successfully: %s' % ent_desc_dict_path)

    return 0


def get_current_upper_bound(dataset_name: str, ea_data_mode: str, tar_ent_list_name: str):
    upper_bound = 0
    model_output_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
        return 0
    filenames = os.listdir(model_output_dir)
    if len(filenames) == 0:
        return 0
    else:
        prefix = ('ent_desc_dict_by_retrieval=%s=%s' % (ea_data_mode, tar_ent_list_name))
        for file_name in filenames:
            suffix = os.path.splitext(file_name)[-1]
            if file_name.startswith(prefix) and suffix == '.json':
                base_name = os.path.splitext(file_name)[0]
                split_list = base_name.split('=')
                cur_upper = int(split_list[-1])
                if upper_bound < cur_upper:
                    upper_bound = cur_upper

    return upper_bound


def retrieve_ent_desc_recursively(dataset_name: str, ea_data_mode: str, tar_ent_list_name: str,
                                  lower_bound: int, upper_bound: int):
    cur_upper = get_current_upper_bound(dataset_name, ea_data_mode, tar_ent_list_name)
    while cur_upper < upper_bound:
        if cur_upper > lower_bound:
            lower_bound = cur_upper

        print('Start to process ent_desc_dict_by_retrieval=%s=%s=%d=%d.json' %
              (ea_data_mode, tar_ent_list_name, lower_bound, upper_bound))
        retrieve_ent_desc(dataset_name, ea_data_mode, tar_ent_list_name, lower_bound, upper_bound)

        cur_upper = get_current_upper_bound(dataset_name, ea_data_mode, tar_ent_list_name)
        time.sleep(1)

    return 0


def complete_ent_desc_by_retrieval(dataset_name: str, ea_data_mode: str, tar_ent_list_name: str):
    ent_desc_file_path = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name,
                                      'ent_desc_dict_by_retrieval=%s=%s.json' %
                                      (ea_data_mode, tar_ent_list_name))

    print('statistics BEFORE completion are:')
    analyze_ent_desc_dict(ent_desc_file_path)

    with open(ent_desc_file_path, 'r', encoding='utf-8') as f:
        ent_desc_dict = json.load(f)

    if dataset_name == 'DBP15K_DE_EN_V1':
        kg = 'dbpedia'
        if tar_ent_list_name == 'ent1':
            ent_lang = 'en'
        elif tar_ent_list_name == 'ent2':
            ent_lang = 'de'
        else:
            raise RuntimeError('Unknown tar_ent_list_name: %s' % tar_ent_list_name)
    else:
        raise RuntimeError('Unknown dataset_name: %s' % dataset_name)

    num_not_valid = 0
    for ent in tqdm(ent_desc_dict):
        desc = ent_desc_dict[ent]
        if not is_desc_valid(desc):
            num_not_valid += 1
            # if isinstance(desc, type(None)):
            #     continue
            print('(%d) retrieving desc for ent "%s" using sparql endpoint' % (num_not_valid, ent))
            desc = call_sparql_dbpedia(ent, ent_lang)
            if not is_desc_valid(desc):
                print('(%d) Desc "%s" is STILL not valid for ent: %s' % (num_not_valid, desc, ent))
            else:
                ent_desc_dict[ent] = desc

    with open(ent_desc_file_path, 'w', encoding='utf-8') as f:
        json.dump(ent_desc_dict, f, ensure_ascii=False, indent=4)
    print('Output generated successfully: %s' % ent_desc_file_path)

    print('statistics AFTER completion are:')
    analyze_ent_desc_dict(ent_desc_file_path)

    return 0


if __name__ == '__main__':
    ## DBP15K_DE_EN_V1, DW15K_V1, DBP15K_FR_EN, DBP15K_JA_EN, DBP15K_ZH_EN
    dataset_name_list = ['DBP15K_DE_EN_V1']
    tar_ent_list_names = ['ent2', 'ent1']
    ea_data_mode = 'all'
    lower_bound = 0
    upper_bound = 100000
    for dataset_name in dataset_name_list:
        for tar_ent_list_name in tar_ent_list_names:
            # retrieve_ent_desc_recursively(dataset_name, ea_data_mode, tar_ent_list_name, lower_bound, upper_bound)
            complete_ent_desc_by_retrieval(dataset_name, ea_data_mode, tar_ent_list_name)
