import numpy as np
from typing import Dict, Set, List
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics(
        query_embeddings: np.array,
        reference_embeddings: np.array,
        ground_truth: List[Set[int]],
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

    ground_truth : List[Set[int]]
        Ground truth labels for the queries, where ground_truth[i] is a set of indices of relevant reference items.

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

    metrics={'precision@1': [], 'precision@5': [], 'recall@1': [], 'recall@5': [], 'AP': [], 'mAP': [] }
    
    number_of_queries=query_embeddings.shape[0]
    if len(ground_truth) != number_of_queries:
        raise ValueError("Length of ground_truth must match number of queries.")
    
    similarities = compute_similarity(query_embeddings, reference_embeddings, similarity_metric)

    metrics['microAP'] = micro_average_precision(similarities, ground_truth)

    for i in range(len(query_embeddings)):
        sorted_indices = np.argsort(similarities[i])[::-1]
        relevant_items = ground_truth[i]
        
        metrics['precision@1'].append(precision_at_k(sorted_indices, relevant_items, k=1))
        metrics['precision@5'].append(precision_at_k(sorted_indices, relevant_items, k=5))
        metrics['recall@1'].append(recall_at_k(sorted_indices, relevant_items, k=1))
        metrics['recall@5'].append(recall_at_k(sorted_indices, relevant_items, k=5))
        metrics['AP'].append(average_precision(sorted_indices, relevant_items))
    
    result = {
        'precision@1': np.mean(metrics['precision@1']),
        'precision@5': np.mean(metrics['precision@5']),
        'recall@1': np.mean(metrics['recall@1']),
        'recall@5': np.mean(metrics['recall@5']),
        'mAP': np.mean(metrics['AP']), 
        'microAP': metrics['microAP']
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

def micro_average_precision(
        similarities: np.array,
        ground_truth: List[Set[int]]
) -> float:
    '''
    Compute micro-Average Precision (µAP) across all query-reference pairs.

    Parameters
    ----------
    similarities : np.array
        Similarity scores between query and reference embeddings.
        Shape: (number_of_queries, number_of_references).
        Higher values indicate greater similarity.

    ground_truth : List[Set[int]]
        Ground truth labels for each query, where ground_truth[i]
        is a set of indices of relevant reference items for query i.

    Returns
    -------
    micro_ap : float
        Micro-Average Precision score, computed as the area under
        the precision-recall curve for all query-reference pairs sorted by confidence.
    '''
    number_of_queries = similarities.shape[0]
    all_pairs = []
    total_relevant = 0
    for query_idx in range(number_of_queries):
        relevant_items = ground_truth[query_idx]
        total_relevant += len(relevant_items)
        for ref_idx in range(similarities.shape[1]):
            all_pairs.append({
                'query_idx': query_idx,
                'ref_idx': ref_idx,
                'similarity': similarities[query_idx, ref_idx],
                'is_relevant': ref_idx in relevant_items
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
    
    return micro_ap


