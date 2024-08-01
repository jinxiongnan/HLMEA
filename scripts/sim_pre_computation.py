import heapq
import json
import os

from torch import nn

from train import tokenize_and_convert_to_tensor, CustomE5Model, CustomMinilmModel, CustomLabseModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging, AutoModel, BertTokenizerFast, \
    BertModel

from train import CustomMPNetModel

logging.set_verbosity_error()
# # Set the CUDA_LAUNCH_BLOCKING environment variable to 1
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# # Set the TORCH_USE_CUDA_DSA environment variable to 1
# os.environ['TORCH_USE_CUDA_DSA'] = '1'


def get_desc_dicts(dataset_name: str, ea_data_mode: str, desc_gen_model: str):
    ent_desc_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_desc_dict', dataset_name)
    if desc_gen_model in ['gpt3.5', 'baichuan2']:
        ent1_desc_file_name = 'ent_desc_dict_by_llm=%s=%s=ent1.json' % \
                              (ea_data_mode, desc_gen_model)
        ent2_desc_file_name = 'ent_desc_dict_by_llm=%s=%s=ent2.json' % \
                              (ea_data_mode, desc_gen_model)
    elif desc_gen_model == 'sparql':
        ent1_desc_file_name = 'ent_desc_dict_by_retrieval=%s=ent1.json' % \
                              (ea_data_mode)
        ent2_desc_file_name = 'ent_desc_dict_by_retrieval=%s=ent2.json' % \
                              (ea_data_mode)
    else:
        raise RuntimeError('Unknown desc_model: %s' % desc_gen_model)

    ent1_desc_file_path = os.path.join(ent_desc_dir, ent1_desc_file_name)
    with open(ent1_desc_file_path, 'r', encoding='utf-8') as f:
        ent1_desc_dict = json.load(f)
    ent2_desc_file_path = os.path.join(ent_desc_dir, ent2_desc_file_name)
    with open(ent2_desc_file_path, 'r', encoding='utf-8') as f:
        ent2_desc_dict = json.load(f)

    ## replace empty desc with ent iri
    for ent in ent1_desc_dict:
        if not is_desc_valid(ent1_desc_dict[ent]):
            ent1_desc_dict[ent] = ent
    for ent in ent2_desc_dict:
        if not is_desc_valid(ent2_desc_dict[ent]):
            ent2_desc_dict[ent] = ent

    return ent1_desc_dict, ent2_desc_dict


def get_triple_dicts(dataset_name: str, ea_data_mode: str, num_triple: int, element_gen_model: str):
    ent_triple_dir = os.path.join(os.getcwd(), '..', 'output', 'ent_triple_dict', dataset_name)
    ent1_triple_file_name = 'ent1_triple=%s=%d=%s.json' % (ea_data_mode, num_triple, element_gen_model)
    ent2_triple_file_name = 'ent2_triple=%s=%d=%s.json' % (ea_data_mode, num_triple, element_gen_model)

    ent1_triple_file_path = os.path.join(ent_triple_dir, ent1_triple_file_name)
    with open(ent1_triple_file_path, 'r', encoding='utf-8') as f:
        ent1_triple_dict = json.load(f)
    ent2_triple_file_path = os.path.join(ent_triple_dir, ent2_triple_file_name)
    with open(ent2_triple_file_path, 'r', encoding='utf-8') as f:
        ent2_triple_dict = json.load(f)

    return ent1_triple_dict, ent2_triple_dict


def is_desc_valid(desc):
    if isinstance(desc, type(None)):
        return False
    elif isinstance(desc, str):
        if desc == 'error':
            return False
        else:
            return True
    else:
        raise RuntimeError('Unknown type for desc: %s' % desc)


def read_groundtruth_with_mode(dataset_name: str, mode: str):
    if dataset_name in ['DBP15K_DE_EN_V1', 'DBP15K_FR_EN_V1', 'DBP100K_DE_EN_V1', 'DBP100K_FR_EN_V1',
                        'DW15K_V1', 'DY15K_V1']:
        file_name = 'ent_links'
    elif dataset_name in ['DBP15K_FR_EN', 'DBP15K_JA_EN', 'DBP15K_ZH_EN']:
        file_name = 'ent_ILLs'
    else:
        raise RuntimeError('Unknown dataset name: %s' % dataset_name)

    file_path = os.path.join(os.getcwd(), '..', 'dataset', dataset_name, file_name)
    if not os.path.exists(file_path):
        raise RuntimeError('Input file "%s" does not exist!' % file_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        if mode == 'train-20':
            mode_lb = 0
            if '15K' in dataset_name:
                mode_ub = 3000
            elif '100K' in dataset_name:
                mode_ub = 20000
            else:
                raise RuntimeError('mode_ub calculation error with mode %s for dataset %s' % (mode, dataset_name))
        elif mode == 'test-80':
            if '15K' in dataset_name:
                mode_lb = 3000
                mode_ub = 15000
            elif '100K' in dataset_name:
                mode_lb = 20000
                mode_ub = 100000
            else:
                raise RuntimeError(
                    'mode_ub calculation error with mode %s for dataset %s' % (mode, dataset_name))
        elif mode == 'all':
            mode_lb = 0
            if '15K' in dataset_name:
                mode_ub = 15000
            elif '100K' in dataset_name:
                mode_ub = 100000
            else:
                raise RuntimeError('mode_ub calculation error with mode %s for dataset %s' % (mode, dataset_name))
        else:
            raise RuntimeError('Unknown mode: %s' % mode)

        index = 0
        gt_dict = {}
        for line in file:
            if index < mode_lb:
                index += 1
                continue
            elif index >= mode_ub:
                break
            else:
                ent1, ent2 = line.strip().split('\t')
                if ent1 not in gt_dict:
                    gt_dict[ent1] = ent2
                else:
                    raise RuntimeError('Multiple aligned entites for %s' % ent1)
                index += 1

    return gt_dict


def get_ent_element(ent: str, ent_element_dict: dict, element_type: str, element_gen_model: str):
    ent_name = ent.split("/")[-1]
    if element_type == 'name':
        ent_element = ent_name
    elif element_type == 'desc':
        if element_gen_model in ['gpt3.5', 'baichuan2', 'sparql']:
            if ent in ent_element_dict.keys():
                ent_element = ent_element_dict[ent]
                if not is_desc_valid(ent_element):
                    ent_element = ent_name
            else:
                print('Warning! Description of %s not contained in ent_desc_dict' % ent)
                ## if desc not available, return name instead
                ent_element = ent_name
        else:
            raise RuntimeError('Unknown desc_model: %s' % element_gen_model)
    elif element_type == 'triple':
        if ent in ent_element_dict.keys():
            ent_triple_list = ent_element_dict[ent]
            ent_element = ''
            for ent_triple in ent_triple_list:
                ent_element += ent_triple + ', '
            ent_element = ent_element[:-2]
        else:
            print('Warning! Triples of %s not contained in ent_triple_dict' % ent)
            ## if triple not available, return empty string instead
            ent_element = ''
    else:
        raise RuntimeError('Unknown element: %s' % element_type)

    return ent_element


# def mean_pooling(embeddings, attention_mask):
#     # Expand attention mask
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
#     # Sum embeddings over sequence length, weighted by the mask
#     sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
#     # Calculate the sum of the mask for each sequence to normalize
#     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#     # Normalize embeddings
#     mean_pooled = sum_embeddings / sum_mask
#     return mean_pooled


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_ptm_model(ptm_model_name: str, tar_sel: str, dataset_name: str, llm_mode: str, triple_sel: str,
                   exp_sel: str, llm_type: str, llm_temp: str, max_rep: int, train_bound: int):
    if 'round' not in ptm_model_name:
        # load original ptm model
        ptm_model = load_original_ptm_model(ptm_model_name)
    else:
        # load fine-tuned ptm model
        # get ptm_model path
        ptm_model_path = get_ptm_model_path(tar_sel, ptm_model_name, dataset_name, llm_mode, triple_sel, exp_sel,
                                            llm_type, llm_temp, max_rep, train_bound)
        # load ptm_model from ptm_model_path
        ptm_model = load_fine_tuned_ptm_model(ptm_model_name, ptm_model_path)

    return ptm_model


def compute_similarity_batch_mpnet_and_minilm(batch_ent1, batch_ent2, device, ptm_model, ptm_model_name):
    if 'round' not in ptm_model_name:
        # load tokenizer
        ptm_tokenizer = load_ptm_tokenizer(ptm_model_name)
        # Tokenize and encode the batch of entity pairs
        inputs1 = ptm_tokenizer(
            batch_ent1, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        inputs2 = ptm_tokenizer(
            batch_ent2, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        # Obtain embeddings
        with torch.no_grad():  # Disable gradient calculation for inference
            embeddings1 = ptm_model(**inputs1)
            embeddings2 = ptm_model(**inputs2)
        # Apply mean pooling to get sentence embeddings
        processed_embeddings1 = mean_pooling(embeddings1, inputs1['attention_mask'])
        processed_embeddings2 = mean_pooling(embeddings2, inputs2['attention_mask'])
    else:
        ptm_model.eval()
        # Obtain embeddings
        input_ids1, attention_mask1 = tokenize_and_convert_to_tensor(ptm_model_name, batch_ent1, device=device)
        input_ids2, attention_mask2 = tokenize_and_convert_to_tensor(ptm_model_name, batch_ent2, device=device)
        with torch.no_grad():
            embeddings1 = ptm_model(input_ids1, attention_mask1, eval=True)
            embeddings2 = ptm_model(input_ids2, attention_mask2, eval=True)
        # Apply mean pooling to get sentence embeddings
        processed_embeddings1 = mean_pooling(embeddings1, attention_mask1)
        processed_embeddings2 = mean_pooling(embeddings2, attention_mask2)

    # Calculate cosine similarities for all pairs
    # Matrix multiplication of embeddings1 with the transpose of embeddings2
    # Resulting matrix is of size batch_size x batch_size
    cosine_similarity_matrix = torch.matmul(processed_embeddings1, processed_embeddings2.transpose(0, 1))

    return cosine_similarity_matrix.cpu().numpy()


def compute_similarity_batch_labse(batch_ent1, batch_ent2, device, ptm_model, ptm_model_name):
    ptm_model.eval()
    if 'round' not in ptm_model_name:
        # load tokenizer
        ptm_tokenizer = load_ptm_tokenizer(ptm_model_name)
        # Tokenize and encode the batch of entity pairs
        inputs1 = ptm_tokenizer(
            batch_ent1, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        inputs2 = ptm_tokenizer(
            batch_ent2, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        # Obtain embeddings
        with torch.no_grad():  # Disable gradient calculation for inference
            embeddings1 = ptm_model(**inputs1)
            embeddings2 = ptm_model(**inputs2)
    else:
        # Obtain embeddings
        input_ids1, attention_mask1 = tokenize_and_convert_to_tensor(ptm_model_name, batch_ent1, device=device)
        input_ids2, attention_mask2 = tokenize_and_convert_to_tensor(ptm_model_name, batch_ent2, device=device)
        with torch.no_grad():
            embeddings1 = ptm_model(input_ids1, attention_mask1, eval=True)
            embeddings2 = ptm_model(input_ids2, attention_mask2, eval=True)
    # Apply pooler_output to get sentence embeddings
    processed_embeddings1 = embeddings1.pooler_output
    processed_embeddings2 = embeddings2.pooler_output

    # Calculate cosine similarities for all pairs
    normalized_embeddings1 = F.normalize(processed_embeddings1, p=2)
    normalized_embeddings2 = F.normalize(processed_embeddings2, p=2)

    return torch.matmul(
        normalized_embeddings1, normalized_embeddings2.transpose(0, 1)
    )


def compute_similarity_batch_e5(batch_ent1, batch_ent2, device, ptm_model, ptm_model_name):
    if 'round' not in ptm_model_name:
        # load tokenizer
        ptm_tokenizer = load_ptm_tokenizer(ptm_model_name)
        # Tokenize and encode the batch of entity pairs
        inputs1 = ptm_tokenizer(
            batch_ent1, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        inputs2 = ptm_tokenizer(
            batch_ent2, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        # Obtain embeddings
        with torch.no_grad():  # Disable gradient calculation for inference
            embeddings1 = ptm_model(**inputs1)
            embeddings2 = ptm_model(**inputs2)
        processed_embeddings1 = average_pool(embeddings1.last_hidden_state, inputs1['attention_mask'])
        processed_embeddings2 = average_pool(embeddings2.last_hidden_state, inputs2['attention_mask'])
    else:
        ptm_model.eval()
        # Obtain embeddings
        input_ids1, attention_mask1 = tokenize_and_convert_to_tensor(ptm_model_name, batch_ent1, device=device)
        input_ids2, attention_mask2 = tokenize_and_convert_to_tensor(ptm_model_name, batch_ent2, device=device)
        with torch.no_grad():
            embeddings1 = ptm_model(input_ids1, attention_mask1, eval=True)
            embeddings2 = ptm_model(input_ids2, attention_mask2, eval=True)
        processed_embeddings1 = average_pool(embeddings1.last_hidden_state, attention_mask1)
        processed_embeddings2 = average_pool(embeddings2.last_hidden_state, attention_mask2)

    # Calculate cosine similarities for all pairs
    normalized_embeddings1 = F.normalize(processed_embeddings1, p=2, dim=1)
    normalized_embeddings2 = F.normalize(processed_embeddings2, p=2, dim=1)

    cosine_similarity_matrix = torch.matmul(normalized_embeddings1, normalized_embeddings2.transpose(0, 1))

    return cosine_similarity_matrix.cpu().numpy()


def load_ptm_tokenizer(ptm_model_name: str):
    if 'e5' in ptm_model_name:
        ptm_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    elif 'labse' in ptm_model_name:
        ptm_tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    elif 'mpnet' in ptm_model_name:
        ptm_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    elif 'minilm' in ptm_model_name:
        ptm_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    else:
        raise RuntimeError('Unknown ptm_model_name: %s' % ptm_model_name)

    return ptm_tokenizer


def load_original_ptm_model(ptm_model_name: str):
    if ptm_model_name == 'e5':
        ptm_model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    elif ptm_model_name == 'labse':
        ptm_model = BertModel.from_pretrained("setu4993/LaBSE")
    elif ptm_model_name == 'mpnet':
        ptm_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    elif ptm_model_name == 'minilm':
        ptm_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    else:
        raise RuntimeError('Unknown ptm_model_name: %s' % ptm_model_name)

    return ptm_model


def get_ptm_model_path(tar_sel: str, ptm_model_name: str, dataset_name: str, llm_mode: str, triple_sel: str,
                       exp_sel: str, llm_type: str, llm_temp: str, max_rep: int, train_bound: int):
    if tar_sel.endswith('sg-ignore'):
        temp_tar_sel = tar_sel[:-10]
        ptm_model_name_splits = ptm_model_name.split('-')
        base_ptm_model_name = ptm_model_name_splits[0]
        ptm_version = int(ptm_model_name_splits[2])
        old_ptm_version = ptm_version - 1
        if old_ptm_version == 0:
            old_ptm_model_name = base_ptm_model_name
        else:
            old_ptm_model_name = '%s-round-%d' % (base_ptm_model_name, old_ptm_version)
        ptm_model_path = os.path.join(os.getcwd(), '..', 'output', 'ptm_model', dataset_name,
                                      'ptm=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.pth' %
                                      (llm_mode, 'train-20', temp_tar_sel, exp_sel, triple_sel,
                                       old_ptm_model_name, llm_type, llm_temp, max_rep, 0, train_bound))
    else:
        ptm_model_path = os.path.join(os.getcwd(), '..', 'output', 'ptm_model', dataset_name,
                                      'ptm=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.pth' %
                                      (llm_mode, 'train-20', tar_sel, exp_sel, triple_sel,
                                       ptm_model_name, llm_type, llm_temp, max_rep, 0, train_bound))

    return ptm_model_path


def load_fine_tuned_ptm_model(ptm_model_name: str, ptm_model_path: str):
    if 'e5' in ptm_model_name:
        ptm_model = CustomE5Model()
    elif 'labse' in ptm_model_name:
        ptm_model = CustomLabseModel()
    elif 'mpnet' in ptm_model_name:
        ptm_model = CustomMPNetModel()
    elif 'minilm' in ptm_model_name:
        ptm_model = CustomMinilmModel()
    else:
        raise RuntimeError('Unknown ptm_model_name: %s' % ptm_model_name)
    try:
        ptm_model.load_state_dict(torch.load(ptm_model_path))
        print(f"Loaded model from {ptm_model_path}")
    except Exception as e:
        print(e)
        raise RuntimeError('Failed to load model from %s' % ptm_model_path)

    return ptm_model


def generate_ptm_sim_dict_batch(dataset_name: str, ea_data_mode: str, ptm_model_name: str, batch_size: int,
                                llm_mode: str, triple_sel: str, exp_sel: str, tar_sel: str, llm_type: str,
                                llm_temp: str, max_rep: int, lower_bound: int, train_bound: int, upper_bound: int):
    ptm_sim_dict = {}
    ptm_sim_dict_file_name = ('ptm_sim_dict=%s=%s=%s=%s=%s=%s=%s=%s=%d=%d=%d.json' %
                              (llm_mode, ea_data_mode, tar_sel, exp_sel, triple_sel, ptm_model_name,
                               llm_type, llm_temp, max_rep, lower_bound, upper_bound))

    ptm_sim_dict_dir = os.path.join(os.getcwd(), '..', 'output', 'ptm_sim_dict', dataset_name)
    if not os.path.exists(ptm_sim_dict_dir):
        os.makedirs(ptm_sim_dict_dir)
    ptm_sim_dict_path = os.path.join(ptm_sim_dict_dir, ptm_sim_dict_file_name)

    ## if ptm_sim_dict exists, load it
    if os.path.exists(ptm_sim_dict_path):
        with open(ptm_sim_dict_path, 'r', encoding='utf-8') as f:
            print('Loading ptm_sim_dict ... ')
            ptm_sim_dict = json.load(f)
        return ptm_sim_dict

    print('Generating ptm_sim_dict ... %s' % ptm_sim_dict_path)
    # Using pre-trained models for similarity calculation
    # Specify the device (use GPU if available)
    device = torch.device(
        "cuda:2" if torch.cuda.device_count() > 2 else "cuda:0" if torch.cuda.is_available() else "cpu")
    # load ptm_model
    ptm_model = load_ptm_model(ptm_model_name, tar_sel, dataset_name, llm_mode, triple_sel, exp_sel, llm_type, llm_temp,
                               max_rep, train_bound)
    # Move the model to the specified device
    ptm_model.to(device)

    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    # Prepare entity lists and descriptions
    ent1_list = list(gt_dict.keys())
    ent2_list = list(gt_dict.values())
    triple_sel_splits = triple_sel.split('-')
    num_triple = int(triple_sel_splits[1])
    element_type = triple_sel_splits[2]
    if element_type == 'desc':
        element_gen_model = triple_sel_splits[3]
        ent1_element_dict, ent2_element_dict = get_desc_dicts(dataset_name, ea_data_mode, element_gen_model)
    elif element_type == 'triple':
        element_gen_model = triple_sel_splits[3]
        ent1_element_dict, ent2_element_dict = get_triple_dicts(
            dataset_name, ea_data_mode, num_triple, element_gen_model)
    else:
        raise RuntimeError('Unknown element_type: %s' % element_type)

    print('Start to process %s with batch size %d' % (ptm_sim_dict_file_name, batch_size))
    # Process in batches
    for i in tqdm(range(0, len(ent1_list), batch_size), position=0,
                  desc="Processing [ent1] batches", leave=True):
        batch_ent1 = [get_ent_element(ent1, ent1_element_dict, element_type, element_gen_model)
                      for ent1 in ent1_list[i:i + batch_size]]
        for j in tqdm(range(0, len(ent2_list), batch_size), position=1,
                      desc="Processing [ent2] batches", leave=False):
            batch_ent2 = [get_ent_element(ent2, ent2_element_dict, element_type, element_gen_model)
                          for ent2 in ent2_list[j:j + batch_size]]

            # Compute similarities for the batch
            if 'e5' in ptm_model_name:
                batch_similarities = compute_similarity_batch_e5(
                    batch_ent1, batch_ent2, device, ptm_model, ptm_model_name)
            elif 'labse' in ptm_model_name:
                batch_similarities = compute_similarity_batch_labse(
                    batch_ent1, batch_ent2, device, ptm_model, ptm_model_name)
            elif ('mpnet' in ptm_model_name) or ('minilm' in ptm_model_name):
                batch_similarities = compute_similarity_batch_mpnet_and_minilm(
                    batch_ent1, batch_ent2, device, ptm_model, ptm_model_name)
            else:
                raise RuntimeError('Unknown ptm_model_name: %s for compute_similarity_batch' % ptm_model_name)

            # Update the similarity dictionary
            for m, ent1 in enumerate(ent1_list[i:i + batch_size]):
                ptm_sim = batch_similarities[m]
                for n, ent2 in enumerate(ent2_list[j:j + batch_size]):
                    # Convert float32 to Python float for JSON serialization
                    similarity_score = float(ptm_sim[n])
                    if ent1 not in ptm_sim_dict:
                        ptm_sim_dict[ent1] = [[ent2, similarity_score]]
                    else:
                        ptm_sim_dict[ent1].append([ent2, similarity_score])

    ## sort in descending order and select top-10 to save
    for ent1 in ptm_sim_dict:
        ent2_sim_pair_list = ptm_sim_dict[ent1]
        # select
        top_k_data_heap = heapq.nlargest(10, ent2_sim_pair_list, key=lambda x: x[1])
        # and sort
        top_k_data_sorted = sorted(top_k_data_heap, key=lambda x: x[1], reverse=True)
        ptm_sim_dict[ent1] = top_k_data_sorted

    ## Save the similarity dictionary
    with open(ptm_sim_dict_path, 'w', encoding='utf-8') as f:
        json.dump(ptm_sim_dict, f, ensure_ascii=False, indent=4)
    print('File generated successfully: %s' % ptm_sim_dict_path)

    # Explicitly delete the model and clear GPU cache
    del ptm_model  # Delete the model from memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear the GPU cache

    return ptm_sim_dict


def is_desc_complete(dataset_name: str, ea_data_mode: str, desc_model: str):
    gt_dict = read_groundtruth_with_mode(dataset_name, ea_data_mode)
    # Prepare entity lists and descriptions
    ent1_list = list(gt_dict.keys())
    ent2_list = list(gt_dict.values())
    ent1_desc_dict, ent2_desc_dict = get_desc_dicts(dataset_name, ea_data_mode, desc_model)

    flag = True
    num_error = 0
    for ent1 in tqdm(ent1_list, desc='ent1 check'):
        if ent1 not in ent1_desc_dict:
            num_error += 1
            print('(%d) ent1 %s not in desc_dict' % (num_error, ent1))
            flag = False

    for ent2 in tqdm(ent2_list, desc='ent2 check'):
        if ent2 not in ent2_desc_dict:
            num_error += 1
            print('(%d) ent2 %s not in desc_dict' % (num_error, ent2))
            flag = False

    return flag


if __name__ == '__main__':
    dataset_name = 'DBP15K_ZH_EN'
    llm_mode = 'icl'
    ea_data_mode = 'test-80'
    triple_sel = 'freq-5'
    exp_sel = ''
    tar_sel = 'ptm-5-triple-freq'
    ptm_model_name = 'labse-round-2'
    llm_type = 'qwen'
    llm_temp = '0'
    max_rep = 5
    lower_bound = 3000
    upper_bound = 15000
    batch_size = 300
    train_bound = 2700

    generate_ptm_sim_dict_batch(dataset_name, ea_data_mode, ptm_model_name, batch_size,
                                llm_mode, triple_sel, exp_sel, tar_sel, llm_type,
                                llm_temp, max_rep, lower_bound, train_bound, upper_bound)
