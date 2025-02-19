from datasets import load_dataset, concatenate_datasets
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes as procrustes
import numpy as np

def _sample_by_topic(dataset_name, topic_column, target_topic, sample_size=100, seed=42, test=False, n_attempts=5):
    """
    Sample rows from a HuggingFace dataset for a specific topic.
    
    Args:
        dataset_name (str): Name of the dataset on HuggingFace
        topic_column (str): Name of the column containing topic information
        target_topic (str): The topic to filter for
        sample_size (int): Number of examples to sample (default: 100)
        seed (int): Random seed for reproducibility (default: 42)
        test (bool): Returns sample of test set if True, otherwise returns sample of train set
        n_attempts (int): Number of attempts to establish connection to HuggingFace (otherwise, potential 504 errors)
    Returns:
        Dataset: A filtered and sampled dataset
    """
    # Load the dataset
    n_attempted = 0
    success=False
    while n_attempted < n_attempts and not success:
        try:
            dataset = load_dataset(dataset_name)
            success=True
        except:
            n_attempted+=1
    
    # Most datasets have a 'train' split by default
    if isinstance(dataset, dict):
        if test:
            dataset = dataset['test']
        else:
            dataset = dataset['train']
    
    # Filter for the target topic
    filtered_dataset = dataset.filter(lambda x: x[topic_column] == target_topic)
    
    # Sample from the filtered dataset
    sampled_dataset = filtered_dataset.shuffle(seed=seed).select(range(min(sample_size, len(filtered_dataset))))
    
    return sampled_dataset


def _merge_datasets(datasets_list, shuffle=True, seed=42):
    """
    Merge multiple datasets into one.
    
    Args:
        datasets_list (list): List of datasets to merge
        shuffle (bool): Whether to shuffle the merged dataset (default: True)
        seed (int): Random seed for shuffling (default: 42)
    
    Returns:
        Dataset: Merged dataset
    """
    # Concatenate all datasets
    merged_dataset = concatenate_datasets(datasets_list)
    
    # Shuffle if requested
    if shuffle:
        merged_dataset = merged_dataset.shuffle(seed=seed)
    
    return merged_dataset


def get_dataset_by_pi(pi, dataset_name, topic_column, target_topics, sample_size=100, seed=42, test=False):
    assert len(target_topics) == len(pi)
    n_topics = len(target_topics)
    
    dataset_size_list = []
    dataset_size = [int(sample_size * pi_component) for pi_component in pi]
    remainder = sample_size - np.sum(dataset_size)
    
    for i in np.random.choice(range(n_topics), size=remainder):
        dataset_size[i] += 1
    
    datasets = [_sample_by_topic(dataset_name, topic_column, tt, dataset_size[i], seed, test) for i, tt in enumerate(target_topics)]
    
    return _merge_datasets(datasets)


def get_query_set(question_key, pi, dataset_name, topic_column, target_topics, sample_size=100, seed=42):
    dataset = get_dataset_by_pi(pi, dataset_name, topic_column, target_topics, sample_size, seed, test=True)
    
    return dataset[question_key]


def _rotate_geometry(true_geometry, to_rotate_geometry):
    R, _ = procrustes(to_rotate_geometry, true_geometry)
    
    return to_rotate_geometry @ R


def _process_text(text, question=False):
    if isinstance(text, list):
        return [_process_text(t, question) for t in text]
    
    while '..' in text:
        text = text.replace('..', '.')
        
    while '!!' in text:
        text = text.replace('!!', '!')
        
    while '!?' in text:
        text = text.replace('!?', '?')
        
    while '.?' in text:
        text = text.replace('.?', '?')
        
    while '??' in text:
        text = text.replace('??', '?')
        
    if question:
        text = text[:text.rfind('?')+1]
        
    return text


def _prepare_dataset(dataset, tokenizer, instruction_key, response_key, max_length=512):
    def __tokenize_function(example):
        question_title = _process_text(example['question_title'], question=True)
        best_answer = _process_text(example['best_answer'])
        
        formatted_text = [f"<|system|>You are a helpful assistant.</s><|user|>{qt}</s><|assistant|>{ba}</s>" for (qt, ba) in zip(question_title, best_answer)]
        
        return tokenizer(
            formatted_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
    
    tokenizer.padding_side='right'
    tokenized_dataset = dataset.map(
        __tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset