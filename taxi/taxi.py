import numpy as np

from graspologic.embed import ClassicalMDS
from .utils import _rotate_geometry
from scipy.spatial.distance import squareform, pdist

import torch
import gc


#- Dataset geometry
def get_dataset_geometry(true_geometry, datasets, embedding_model):
    '''
    Returns the geometry estimated by embedding the question / answer pairs in datasets, rotated to fit true_geometry.
    '''
    
    embeddings = {}
    
    for vertex in datasets:
        N = len(datasets[vertex])
        
        questions = datasets[vertex]['question_title']
        answers = datasets[vertex]['best_answer']
        documents =[]
        for i in range(N):
            documents.append(questions[i] + ' ' + answers[i])
        embeddings[vertex] = embedding_model.encode(documents)
                        
    mean_embeddings = np.array([np.mean(embeddings[v], axis=0) for v in embeddings])
    dist_dataset = squareform(pdist(mean_embeddings))
    dist_dataset /= np.linalg.norm(dist_dataset)
    dataset_geometry = ClassicalMDS(n_components=true_geometry.shape[1]).fit_transform(dist_dataset)
        
    return _rotate_geometry(true_geometry, dataset_geometry)


#- Structural geometry
def get_lora_matrices(model, target_modules = ["q_proj","k_proj","v_proj"]):
    '''
    Returns the lora matrices (A, B) corresponding to each layer and each module.
    '''
    
    if 'base_model' in model._modules:
        layers = model._modules['base_model']._modules['model']._modules['model']._modules['layers']
    else:
        model._modules['model']._modules['layers']

    lora_matrices = []

    for ell in layers:
        attention_layers = ell._modules['self_attn']
        for target_module in target_modules:
            target = attention_layers._modules[target_module]._modules
            lora_A=target['lora_A']._modules['default'].weight.data.to('cpu').numpy()
            lora_B=target['lora_B']._modules['default'].weight.data.to('cpu').numpy()

            lora_matrices.append(lora_A)
            lora_matrices.append(lora_B)

    return lora_matrices


def get_structural_geometry(true_geometry, lora_matrices):
    '''
    Returns the geometry estimated via cmds of the lora matrices, rotated to fit true_geometry.
    '''
    
    vertices = list(lora_matrices.keys())
    distance_structure = np.zeros((len(vertices), len(vertices)))

    for i, vertex1 in enumerate(vertices):
        lora_matrices1 = lora_matrices[vertex1]
        for j, vertex2 in enumerate(vertices[i+1:], i+1):
            temp_diff = 0

            for ii, matrix in enumerate(lora_matrices1):
                temp_diff += np.linalg.norm(matrix - lora_matrices[vertex2][ii])**2

            temp_diff = np.sqrt(temp_diff)
            distance_structure[i,j] = temp_diff
            distance_structure[j,i] = temp_diff


    distance_structure /= np.linalg.norm(distance_structure)
    structure_geometry = ClassicalMDS(n_components=true_geometry.shape[1]).fit_transform(distance_structure)
    return _rotate_geometry(true_geometry, structure_geometry)


#- Functional and behavioral geometry
def get_outputs(model, tokenizer, user_content_list, max_length=32, match_n_input_tokens=False):
    '''
    Returns responses and normed average last hidden states.
    '''
    
    message_list = []
    system_prompt = 'You are a helpful assistant.'

    for user_content in user_content_list:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""}
        ]

        message_list.append(messages)

    formatted_text = tokenizer.apply_chat_template(message_list, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(formatted_text, 
                       return_tensors="pt", 
                       padding=True,
                       padding_side='left',
                       max_length=64
    )

    inputs.to('cuda:0')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
        inputs.to('cpu')
        undressed_inputs = tokenizer(user_content_list, return_tensors="pt", 
                   padding=True,
                   padding_side='left',
                   max_length=64).to('cuda:0')
        
        processed_inputs = model(undressed_inputs['input_ids'], output_hidden_states=True)
        undressed_inputs.to('cpu')

    # Extract the last hidden state
    last_hidden_states = processed_inputs.hidden_states[-1].to('cpu')
            
    if match_n_input_tokens:
        n_tokens_list = undressed_inputs['attention_mask'].sum(dim=-1)
    else:
        n_tokens_list = None
    
    responses = _get_responses(tokenizer, outputs, inputs, n_tokens_list)
    return responses, _get_normed_average_last_state(last_hidden_states, undressed_inputs)


def _get_normed_average_last_state(last_hidden_states, inputs):
    embds = []

    for i, last_state_sequence in enumerate(last_hidden_states): 
        avg = []
        for ii, last_state in enumerate(last_state_sequence):
            if inputs['attention_mask'][i, ii]:
                avg.append(last_state.numpy())

        avg = np.mean(avg, axis=0)
        avg /= np.linalg.norm(avg)

        embds.append(avg)
        
    return np.array(embds)


def _get_responses(tokenizer, outputs, inputs, n_tokens_list=None):
    if n_tokens_list is None:
        n_tokens_list = [1000 for o in outputs[0]]
        
    to_decode_list = []
    for i,response_token_list in enumerate(outputs[0]):
        n_response_tokens = len(response_token_list)
        n_tokens = n_tokens_list[i]
                
        last_asst_id = n_response_tokens - list(response_token_list)[-1::-1].index(78191) - 1
        to_decode_list.append(response_token_list[last_asst_id + 3: last_asst_id + 3 + n_tokens])
        
    response_string_list = tokenizer.batch_decode(to_decode_list, skip_special_tokens=True)
    
    return response_string_list


def get_functional_geometry(true_geometry, embeddings):
    '''
    Returns the geometry estimated via cmds of last hidden states, rotated to fit true_geometry.
    '''
    dist_functional = np.zeros((len(embeddings), len(embeddings)))
    
    vertices = list(embeddings.keys())
    for i, vertex1 in enumerate(vertices):
        for ii, vertex2 in enumerate(vertices[i+1:], i+1):
            dist_functional[i,ii] = np.linalg.norm(embeddings[vertex1] - embeddings[vertex2])
            dist_functional[ii,i] = dist_functional[i,ii]
            
    dist_functional /= np.linalg.norm(dist_functional)
    functional_geometry = ClassicalMDS(n_components=true_geometry.shape[1]).fit_transform(dist_functional)
    
    return _rotate_geometry(true_geometry, functional_geometry)
        

def get_behavioral_geometry(true_geometry, responses, embedding_model):
    '''
    Returns the geometry estimated via DKPS, rotated to fit true_geometry.
    '''
    embeddings = {}
    
    for vertex in responses:
        embeddings[vertex] = embedding_model.encode(responses[vertex])
        
    dist_behavioral = np.zeros((len(embeddings), len(embeddings)))
    
    vertices = list(embeddings.keys())
    for i, vertex1 in enumerate(vertices):
        for ii, vertex2 in enumerate(vertices[i+1:], i+1):
            dist_behavioral[i,ii] = np.linalg.norm(embeddings[vertex1] - embeddings[vertex2])
            dist_behavioral[ii,i] = dist_behavioral[i,ii]
            
    dist_behavioral /= np.linalg.norm(dist_behavioral)
    behavioral_geometry = ClassicalMDS(n_components=true_geometry.shape[1]).fit_transform(dist_behavioral)
    
    return _rotate_geometry(true_geometry, behavioral_geometry)