import json

from huggingface_hub import login
# api_key_path = "/home/paperspace/api_keys.json"
# with open(api_key_path, 'r') as j:
#     key = json.loads(j.read())['hf-llama']

#login(token=key)
    
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from graspologic.embed import ClassicalMDS

from sentence_transformers import SentenceTransformer
from datasets import load_dataset, concatenate_datasets

from tqdm import tqdm
from taxi import utils, taxi

EMBEDDING_MODEL = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device='cuda:0')


import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)

import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes as procrustes


model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
tokenizer.padding_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='right'


def get_fine_pis(coarse_pi, coarse_to_fine_dict):
    fine_topics = []
    fine_pi = []
    
    for i, coarse in enumerate(coarse_to_fine_dict):
        for fine in coarse_to_fine_dict[coarse]:
            fine_topics.append(fine)
            fine_pi.append(coarse_pi[coarse] / len(coarse_to_fine_dict[coarse]))
            
    return tuple(fine_topics), tuple(fine_pi)


def pi_to_string(pi):
    s=""
    for c in pi[:-1]:
        s+= str(int(100*c // 1)).zfill(3) + '_'
        
    s+=str(int(100 * pi[-1] // 1)).zfill(3)
    
    return s

TRAIN=True

# group 1: Health (2), Sports (5), Family (8)
# group 2: Politics (9), business (6), society and culture (0)
# group 3: Computers (4), Entertainment (7), Education (3) 
coarse_to_fine_dict = {
    0: (2,5,8),
    1: (9,6,0),
    2: (4,7,3)
}

dataset_name = 'community-datasets/yahoo_answers_topics'
topic_key = 'topic'

'''
lora_rank_list = [8, 16, 32, 128]
N_list = [10,50,100,500,1000,5000,10000]
n_mc=5
'''
lora_rank_list = [8, 32, 128]
N_list = [10,100,1000,10000]
n_mc=1


cache_dir = '~/.cache/huggingface/datasets'
for N in N_list:
    N_str = str(N).zfill(5)
    
    for lora_rank in lora_rank_list:
        print(f"training: N={N}, lora_rank={lora_rank}")
                
        lora_config = LoraConfig(
            r=lora_rank,  # Reduced rank for 8B model
            lora_alpha=2*lora_rank,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj"
            ]
        )
        
        lora_rank_str = str(lora_rank).zfill(3)
        for mc in range(n_mc):
            datasets = {}
            # lora_matrices = {}
            # normed_average_hidden_states = {}
            # responses = {}
            
            ### Update true_pi_list here if necessary
            true_pi_list = [(1,0,0), (0,1,0), (0,0,1), (1/3,1/3,1/3)]
                                                       
            for true_pi in true_pi_list:
                fine_topics, fine_pi = get_fine_pis(true_pi, coarse_to_fine_dict)
                
                save_string = f'./models/N_{N_str}_lora_{lora_rank_str}_id_{mc}_pi_{pi_to_string(true_pi)}'
                if os.path.exists(save_string) and TRAIN:
                    continue
                
                datasets[true_pi] = utils.get_dataset_by_pi(fine_pi, dataset_name, topic_key, fine_topics, N, cache_dir=cache_dir, seed=mc)
                
                if 'base_model' not in locals():
                    base_model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16,
                            device_map="cuda:0",
                            trust_remote_code=True
                    )
                
                if TRAIN:
                    tokenized_dataset = utils._prepare_dataset(datasets[true_pi], tokenizer, 'question_title', 'best_answer')
                    
                    # Prepare model for training
                    model = prepare_model_for_kbit_training(base_model)
                    model = get_peft_model(model, lora_config)
                    
                    # Initialize trainer
                    training_args = TrainingArguments(
                        output_dir=save_string,
                        num_train_epochs=3,
                        per_device_train_batch_size=4,
                        gradient_accumulation_steps=1,
                        learning_rate=1e-5,  
                        fp16=True,
                        save_strategy='no',
                        logging_steps=10,
                        optim="paged_adamw_8bit",
                        lr_scheduler_type="cosine"
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_dataset,
                        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                    )

                    # Start training
                    trainer.train()
                    model.save_pretrained(save_string)
                else:
                    # Load base model
                    model = PeftModel.from_pretrained(
                                base_model,
                                save_string,
                                torch_dtype=torch.float16,
                                device_map='cuda:0'
                    )
                                

                # responses[true_pi], normed_average_hidden_states[true_pi] = taxi.get_outputs(model, tokenizer, query_set, match_n_input_tokens=True)
                # lora_matrices[true_pi] = taxi.get_lora_matrices(model)

                            
                try:
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                except:
                    pass

            # cached_data[N][lora_rank][mc]['behavioral'] = responses.copy()
            # cached_data[N][lora_rank][mc]['functional'] = normed_average_hidden_states.copy()
            # cached_data[N][lora_rank][mc]['structural'] = lora_matrices.copy()

            
            # pickle.dump(cached_data, open(cache_file_path, 'wb'))
        
try:
    base_model.cpu()
    del base_model
    torch.cuda.empty_cache()
    gc.collect()
except:
    pass