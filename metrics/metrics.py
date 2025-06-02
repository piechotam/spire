import h5py
import numpy as np
from typing import Dict, List, Set, Optional, Literal
from collections import namedtuple
from sklearn.metrics.pairwise import cosine_similarity
from math import ceil

def create_ground_truth_dict(path_to_gt_file: str) -> Dict[int, int]:
    '''
    Creates ground truth dict where ground_truth[idx] is the index of the relevant reference item for query with index idx.

    Parameters
    ----------
    path_to_gt_file : str
        Path to csv file with ground truth.
    
    Returns
    -------
    ground_truth : Dict[int, int]
        Ground truth mapping where ground_truth[idx] is the index of the relevant reference item for query with index idx.
    '''
    ground_truth = dict()
    with open(path_to_gt_file, "r") as f:
        for match in f.readlines():
            query, reference = [int(name[1:]) for name in match.split(",")]
            ground_truth[query] = reference
    return ground_truth

def compute_similarity(
        query_embeddings: np.array, 
        reference_embeddings: np.array, 
        metric: str
) -> np.array:
    '''
    Compute the similarity between query and reference embeddings.

    Parameters
    ----------
    query_embeddings : np.array
        Embeddings of the queries.

    reference_embeddings : np.array
        Embeddings of the reference items.

    metric : str
        The similarity metric to use (cosine' for CLIP, 'manhattan' for SAE).

    Returns
    -------
    similarities : np.array
        Similarity scores between query and reference embeddings.
        For 'cosine', higher values mean more similar.
        For 'manhattan', negative distances are returned (higher values mean more similar).
    '''
    
    if metric == 'cosine':
        return cosine_similarity(query_embeddings, reference_embeddings)
    elif metric == 'manhattan':
        diff = np.abs(query_embeddings[:, np.newaxis, :] - reference_embeddings)
        distances = np.sum(diff, axis=2)
        return -distances
    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'manhattan'.")


def precision_at_k(
        sorted_indices: np.ndarray, 
        relevant_items: Set[int],
        k: int
) -> float:
    '''
    Compute precision at K.
    
    Parameters
    ----------
    sorted_indices : np.ndarray
        Indices of reference items sorted by similarity (descending).
    relevant_items : Set[int]
        Set of indices of truly relevant items.
    k : int
        Number of top items to consider.
    
    Returns
    -------
    float
        Precision@K score.
    '''
    top_k = sorted_indices[:k]
    if len(top_k) == 0:
        return 0.0
    TP = len(set(top_k) & relevant_items)
    return TP / len(top_k)

def recall_at_k(
        sorted_indices: np.ndarray, 
        relevant_items: Set[int], 
        k: int
) -> float:
    '''
    Compute recall at K.
    
    Parameters
    ----------
    sorted_indices : np.ndarray
        Indices of reference items sorted by similarity (descending).
    relevant_items : Set[int]
        Set of indices of truly relevant items.
    k : int
        Number of top items to consider.
    
    Returns
    -------
    float
        Recall@K score.
    '''
    top_k = sorted_indices[:k]
    if len(relevant_items) == 0:
        return 0.0
    TP = len(set(top_k) & relevant_items)
    FN = len(relevant_items - set(top_k))
    return TP / (TP + FN)

def _process_single_reference(
        matched_pairs: dict,
        reference_embeddings_path: str,
        model: str,
        model_embeddings_filename: str,
        indices_filename: str,
        query_embeddings_fileref: h5py.Dataset,
        query_indices_fileref: h5py.Dataset,
        batch_size: int
):
    '''
    Processes single reference set by finding nearest neighbor embeddings from that set.
    '''
    RefMatch = namedtuple('RefMatch', 'reference_idx, similarity')
    metric = 'cosine' if model == 'clip' else 'manhattan'
    query_embeddings_position = 0
    n_queries = len(query_embeddings_fileref)
    n_batches = ceil(n_queries / batch_size)

    with h5py.File(f"{reference_embeddings_path}/{model_embeddings_filename}") as reference_embeddings_file, \
         h5py.File(f"{reference_embeddings_path}/{indices_filename}") as reference_indices_file:
        reference_embeddings = reference_embeddings_file[f"{model}_embeddings"][:] # at most 6GB loaded to RAM in our case, speeds up calculation
        reference_indices = reference_indices_file["image_indices"]
        for query_batch_number in range(n_batches):
            query_embeddings_batch = query_embeddings_fileref[query_embeddings_position:(query_embeddings_position + 1) * batch_size]
            similarities = compute_similarity(query_embeddings_batch, reference_embeddings, metric)
            nearest_neighbors_array_indices = np.argmax(similarities, axis=1)
            nearest_neighbors_reference_indices = reference_indices[nearest_neighbors_array_indices]
            
            for i, query_index in enumerate(query_indices_fileref[query_embeddings_position:(query_embeddings_position + 1) * batch_size]):
                ref_match_idx = nearest_neighbors_reference_indices[i]
                similarity = similarities[i, nearest_neighbors_array_indices[i]]
                current_query_ref_match = matched_pairs.get(query_index)
                new_query_ref_match = RefMatch(ref_match_idx, similarity)

                if current_query_ref_match is None:
                    matched_pairs[query_index] = new_query_ref_match
                elif current_query_ref_match.similarity < similarity:
                    matched_pairs[query_index] = new_query_ref_match

            query_embeddings_position += batch_size

def create_matched_pairs(
        query_embeddings_path: str,
        reference_embeddings_paths: List[str],
        model: Literal['clip', 'sae'],
        batch_size: int):
    '''
    Finds nearest neighbor referenece embedding for each query embedding.
    '''
    matched_pairs = dict()
    model_embeddings_filename = model + '.h5'
    indices_filename = 'indicies.h5' # there was a typo while saving files - using the same name here

    with h5py.File(f"{query_embeddings_path}/{model_embeddings_filename}") as query_embeddings_file, \
         h5py.File(f"{query_embeddings_path}/{indices_filename}") as query_indices_file:
        query_embeddings_fileref = query_embeddings_file[f"{model}_embeddings"]
        query_indices_fileref = query_indices_file["image_indices"]
        
        for reference_embeddings_path in reference_embeddings_paths:
            print(f"Processing queries with references from {reference_embeddings_path}...")
            _process_single_reference(
                matched_pairs=matched_pairs,
                reference_embeddings_path=reference_embeddings_path,
                model=model,
                model_embeddings_filename=model_embeddings_filename,
                indices_filename=indices_filename,
                query_embeddings_fileref=query_embeddings_fileref,
                query_indices_fileref=query_indices_fileref,
                batch_size=batch_size
            )

    return matched_pairs

def micro_average_precision(
        matched_pairs: List[Dict],
        number_of_positives: int
):
    '''
    Calculates the microAP. Saves precision and recall for different thresholds.
    '''
    matched_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    micro_ap = 0.0
    num_relevant_found = 0
    prev_recall = 0.0
    precision_recall_history = np.empty(shape=(len(matched_pairs), 3))

    for i, pair in enumerate(matched_pairs):
        if pair['is_relevant']:
            num_relevant_found += 1
        precision = num_relevant_found / (i + 1)
        recall = num_relevant_found / number_of_positives
        delta_recall = recall - prev_recall
        micro_ap += precision * delta_recall
        prev_recall = recall
        threshold = pair['similarity']
        precision_recall_history[i] = [precision, recall, threshold]
    
    return micro_ap, precision_recall_history

def average_precision(
        sorted_indices: np.ndarray, 
        relevant_items: Set[int]
) -> float:
    '''
    Compute Average Precision (AP).
    
    Parameters
    ----------
    sorted_indices : np.ndarray
        Indices of reference items sorted by similarity (descending).
    relevant_items : Set[int]
        Set of indices of truly relevant items.
    
    Returns
    -------
    float
        Average Precision score.
    '''
    if not relevant_items:
        return 0.0
    
    ap, num_relevant = 0.0, 0
    for k in range(1, len(sorted_indices) + 1):
        if sorted_indices[k-1] in relevant_items:
            num_relevant += 1
            precision = num_relevant / k
            ap += precision
    
    ap /= len(relevant_items)
    return ap

def compute_metrics(
        query_embeddings: np.array,
        reference_embeddings: np.array,
        ground_truth: Dict[int, int],
        similarity_metric: str='cosine'
) -> Dict[str, float]:
    '''
    Compute metrics for evaluating the performance of a model based on query and reference embeddings.

    Parameters
    ----------
    query_embeddings : np.array
        Embeddings of the queries.

    reference_embeddings : np.array
        Embeddings of the reference items.

    ground_truth : Dict[int, int]
        Ground truth mapping where ground_truth[idx] is the index of the relevant reference item for query with index idx.

    similarity_metric : str
        The similarity metric to use (cosine' for CLIP, 'manhattan' for SAE).

    Returns
    -------

    metrics : Dict[str, float]
       A dictionary containing the computed metrics:
            - precision@1: Average precision for top-1 retrieved items.
            - precision@5: Average precision for top-5 retrieved items.
            - recall@1: Average recall for top-1 retrieved items.
            - recall@5: Average recall for top-5 retrieved items.
            - mAP: Mean Average Precision across all queries.
            - microAP: Micro-Average Precision across all queries.
    '''

    metrics={'precision@1': [], 'precision@5': [], 'recall@1': [], 'recall@5': [], 'AP': []}
    
    all_pairs = []
    total_relevant = 0

    for i in range(len(query_embeddings)):
        relevant_item = ground_truth.get(i)
        if relevant_item is None:
            continue

        query = query_embeddings[i:i+1]
        similarities = compute_similarity(query, reference_embeddings, similarity_metric)[0]
        relevant_items = {relevant_item}
        sorted_indices = np.argsort(similarities)[::-1]
        
        metrics['precision@1'].append(precision_at_k(sorted_indices, relevant_items, k=1))
        metrics['precision@5'].append(precision_at_k(sorted_indices, relevant_items, k=5))
        metrics['recall@1'].append(recall_at_k(sorted_indices, relevant_items, k=1))
        metrics['recall@5'].append(recall_at_k(sorted_indices, relevant_items, k=5))
        metrics['AP'].append(average_precision(sorted_indices, relevant_items))

        total_relevant += 1
        for ref_idx, sim in enumerate(similarities):
            all_pairs.append({
                'query_idx': i,
                'ref_idx': ref_idx,
                'similarity': sim,
                'is_relevant': (ref_idx == relevant_item)
            })

    all_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    micro_ap = 0.0
    num_relevant_found = 0
    prev_recall = 0.0
    
    for i, pair in enumerate(all_pairs, 1):
        if pair['is_relevant']:
            num_relevant_found += 1
            precision = num_relevant_found / i
            recall = num_relevant_found / total_relevant if total_relevant > 0 else 0.0
            delta_recall = recall - prev_recall
            micro_ap += precision * delta_recall
            prev_recall = recall

    
    result = {
        'precision@1': np.mean(metrics['precision@1']),
        'precision@5': np.mean(metrics['precision@5']),
        'recall@1': np.mean(metrics['recall@1']),
        'recall@5': np.mean(metrics['recall@5']),
        'mAP': np.mean(metrics['AP']), 
        'microAP': micro_ap
    }
    
    return result