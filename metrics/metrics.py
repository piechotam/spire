import numpy as np
from typing import Dict, Set, List, Optional
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics(
        query_embeddings: np.array,
        reference_embeddings: np.array,
        ground_truth: Dict[int, Optional[int]],
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

    ground_truth : Dict[int, Optional[int]]
        Ground truth mapping where ground_truth[i] is the index of the relevant reference item for query i, or None if there is no match.

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
    
    number_of_queries=query_embeddings.shape[0]
    if len(ground_truth) != number_of_queries:
        raise ValueError("Length of ground_truth must match number of queries.")
    
    all_pairs = []
    total_relevant = 0

    for i in range(len(query_embeddings)):
        relevant_item = ground_truth.get(i)
        if relevant_item is None:
            continue

        query = query_embeddings[i:i+1]
        similarities = compute_similarity(query, reference_embeddings, similarity_metric)[0] #shape (1, n_references)
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

    #micro average precision
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
