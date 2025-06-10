import h5py
import pickle
import numpy as np
import faiss
from typing import Dict, List, Set, Tuple, Literal
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from math import ceil

REFERENCE_SET_SIZE = 50000

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
        metric: Literal['cosine', 'manhattan']
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
        distances = manhattan_distances(query_embeddings, reference_embeddings)
        return -distances
    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'manhattan'.")


def precision_top_k(
        nearest_neighbors: Dict[int, List[Tuple[float, int, bool]]], 
        k: int,
        threshold: float
) -> float:
    '''
    Compute top_k precision.
    
    Parameters
    ----------
    nearest_neighbors: Dict[int, List[Tuple[float, int, bool]]]
        Dictonary with query indices as keys and lists with k_nearest neighbors reference index and similarity as values.
    k : int
        Number of nearest neighbors to consider.
    threshold: float
        Classification threshold.
        
    Returns
    -------
    float
        Top_k precision.
    '''
    predicted_positive = true_positive = 0
    for query_idx, nn_list in nearest_neighbors.items():
        if len(nn_list) < k:
            raise ValueError(f"Only {len(nn_list)} nearest neighbors provided. Got k = {k}.")
        top_k_nn = nn_list[:k]
        for nearest_neighbor in top_k_nn:
            nn_similarity = nearest_neighbor[0]
            correct_match = nearest_neighbor[2]
            if nn_similarity > threshold:
                predicted_positive += 1
                if correct_match:
                    true_positive += 1

    return true_positive / predicted_positive if predicted_positive > 0 else 0.0

def recall_top_k(
        nearest_neighbors: Dict[int, List[Tuple[float, int, bool]]],
        k: int,
        threshold: float,
        n_positives: int
) -> float:
    '''
    Compute top_k recall.
    
    Parameters
    ----------
    nearest_neighbors: Dict[int, List[Tuple[float, int, bool]]]
        Dictonary with query indices as keys and lists with k_nearest neighbors reference index and similarity as values.
    k : int
        Number of nearest neighbors to consider.
    threshold: float
        Classification threshold.
    n_positives: int
        Number of positives. 
    
    Returns
    -------
    float
        Top_k recall.
    '''
    true_positive = 0
    for query_idx, nn_list in nearest_neighbors.items():
        if len(nn_list) < k:
            raise ValueError(f"Only {len(nn_list)} nearest neighbors provided. Got k = {k}.")
        top_k_nn = nn_list[:k]
        for nearest_neighbor in top_k_nn:
            nn_similarity = nearest_neighbor[0]
            correct_match = nearest_neighbor[2]
            if nn_similarity > threshold and correct_match:
                    true_positive += 1

    return true_positive / n_positives if n_positives > 0 else 0.0

def _filter_top_k_activations(
        sae_embeddings: np.ndarray,
        top_k):
    valid_top_k = min(top_k, sae_embeddings.shape[1])
    kth_values = np.partition(sae_embeddings, -valid_top_k, axis=1)[:, -valid_top_k]
    sae_embeddings[sae_embeddings < kth_values[:, np.newaxis]] = 0

def _process_single_reference(
        nearest_neighbors: Dict[int, List[Tuple[float, int, bool]]],
        reference_embeddings_path: str,
        model: str,
        model_embeddings_filename: str,
        ground_truth: Dict[int, int],
        indices_filename: str,
        query_embeddings_fileref: h5py.Dataset,
        query_indices_fileref: h5py.Dataset,
        n_queries: int,
        k_nearest: int,
        batch_size: int,
        top_k: int | None = None
):
    '''
    Processes single reference set by finding k_nearest neighbors embeddings from that set.
    '''
    metric = 'cosine' if model == 'clip' else 'manhattan'
    query_embeddings_position = 0
    n_queries = min(n_queries, len(query_embeddings_fileref))
    n_batches = ceil(n_queries / batch_size)

    with h5py.File(f"{reference_embeddings_path}/{model_embeddings_filename}") as reference_embeddings_file, \
         h5py.File(f"{reference_embeddings_path}/{indices_filename}") as reference_indices_file:
        reference_embeddings = reference_embeddings_file[f"{model}_embeddings"][:]
        
        if model == "sae" and top_k is not None:
            print(f"Applying top_k={top_k} filtering to SAE reference embeddings...")
            _filter_top_k_activations(reference_embeddings, top_k)
            print("Finished applying top_k filtering.")

        reference_indices = reference_indices_file["image_indices"][:]
        k_nearest = min(k_nearest, reference_embeddings.shape[0])
        
        if model == "sae":
            faiss_index = faiss.IndexFlat(reference_embeddings.shape[1], faiss.METRIC_L1)
            try:
                faiss_index.add(reference_embeddings)
                print(f"Faiss L1 index built successfully. Index total: {faiss_index.ntotal}")
            except Exception as e:
                print(f"Error building Faiss L1 index: {e}")
                faiss_index = None

        for batch_number in range(n_batches):
            start_idx = query_embeddings_position
            end_idx = min(query_embeddings_position + batch_size, n_queries)
            if start_idx >= end_idx: break
            query_embeddings_batch = query_embeddings_fileref[start_idx:end_idx]
            
            if model == "sae" and top_k is not None:
                _filter_top_k_activations(query_embeddings_batch, top_k)

            if model == "sae" and faiss_index is not None:
                query_embeddings_batch_faiss = np.ascontiguousarray(query_embeddings_batch.astype('float32'))
                distances, nearest_neighbors_array_indices_k_nearest = faiss_index.search(query_embeddings_batch_faiss, k_nearest)
                similarities = -distances
            else:
                similarities = compute_similarity(query_embeddings_batch, reference_embeddings, metric)
                nearest_neighbors_array_indices_k_nearest = np.argpartition(similarities, -k_nearest, axis=1)[:, -k_nearest:]
            
            for i, query_index in enumerate(query_indices_fileref[start_idx:end_idx]):
                query_k_nearest_array_indices = nearest_neighbors_array_indices_k_nearest[i]
                query_k_nearest_reference_indices = reference_indices[query_k_nearest_array_indices]
                
                if model == "clip" or (model == "sae" and faiss_index is None):
                    query_k_similarities = similarities[i, query_k_nearest_array_indices]
                else:
                    query_k_similarities = similarities[i]
                
                query_ground_truth = ground_truth.get(query_index)
                found_k_nearest = [(sim, ref_idx, ref_idx == query_ground_truth) for sim, ref_idx in zip(query_k_similarities, query_k_nearest_reference_indices)]
                current_k_nearest = nearest_neighbors.get(query_index, [])
                current_k_nearest.extend(found_k_nearest)
                current_k_nearest.sort(key=lambda t: t[0], reverse=True)
                nearest_neighbors[query_index] = current_k_nearest[:k_nearest]
                
            query_embeddings_position += batch_size

def find_nearest_neighbors(
        query_embeddings_path: str,
        reference_embeddings_paths: List[str],
        model: Literal['clip', 'sae'],
        ground_truth: Dict[int, int],
        n_queries: int,
        k_nearest: int,
        batch_size: int,
        save_path: str | None = None,
        top_k: int | None = None
        ):
    '''
    Finds nearest neighbor referenece embedding for each query embedding.

    Parameters
    ----------
    query_embeddings_path: str
        Path to directory with query embeddings.
    
    reference_embeddings_paths: List[str]
        List of paths to directories with reference embeddings.
    
    model: Literal['clip', 'sae']
        Which model embeddings to use.
    
    batch_size: int
        Batch size of queries for similarity calculation.

    save_path: str | None = None 
        Path to where the nearest_neighbors will be saved. If None the output will not be saved.
    
    Returns
    -------
    nearest_neighbors: Dict[int, List[Tuple[float, int, bool]]]
        Dictonary with query indices as keys and lists with k_nearest neighbors reference index and similarity as values.
    '''
    nearest_neighbors = dict()
    model_embeddings_filename = model + '.h5'
    indices_filename = 'indicies.h5' # there was a typo while saving files - using the same name here

    with h5py.File(f"{query_embeddings_path}/{model_embeddings_filename}") as query_embeddings_file, \
         h5py.File(f"{query_embeddings_path}/{indices_filename}") as query_indices_file:
        query_embeddings_fileref = query_embeddings_file[f"{model}_embeddings"]
        query_indices_fileref = query_indices_file["image_indices"]
        
        for reference_embeddings_path in reference_embeddings_paths:
            print(f"Processing queries with references from {reference_embeddings_path}...")
            _process_single_reference(
                nearest_neighbors=nearest_neighbors,
                reference_embeddings_path=reference_embeddings_path,
                model=model,
                model_embeddings_filename=model_embeddings_filename,
                ground_truth=ground_truth,
                indices_filename=indices_filename,
                query_embeddings_fileref=query_embeddings_fileref,
                query_indices_fileref=query_indices_fileref,
                n_queries=n_queries,
                k_nearest=k_nearest,
                batch_size=batch_size,
                top_k=top_k
            )
    
    if save_path is not None:
        print(f"Saving nearest_neighbors to {save_path}/nearest_neighbors.pkl...")
        with open(f"{save_path}/nearest_neighbors.pkl", 'wb') as f:
            pickle.dump(nearest_neighbors, f)

    return nearest_neighbors

def micro_average_precision(
        nearest_neighbors: Dict[int, List[Tuple[float, int, bool]]],
        number_of_positives: int,
        save_path: str | None = None
):
    '''
    Calculates the microAP. Saves precision and recall for different thresholds.

    Parameters
    ----------
    nearest_neighbors: Dict[int, List[Tuple[float, int, bool]]]
        Dictonary with query indices as keys and lists with k_nearest neighbors reference index and similarity as values.
        For this calculation only top1 nearest neighbors are needed.
    
    number_of_positives: int
        Number of positive samples.
    
    save_path: str | None = None 
        Path to where the output will be saved. If None the output will not be saved.
    
    Returns
    -------
    micro_ap: float
        The microAP value.
    
    precision_recall_history: np.ndarray
        A numpy array with precision-recall pairs for different threshold values.
    '''
    nearest_neighbors = sorted(nearest_neighbors.items(), key=lambda item: item[1][0], reverse=True)
    micro_ap = 0.0
    num_relevant_found = 0
    prev_recall = 0.0
    precision_recall_history = np.empty(shape=(len(nearest_neighbors), 3))

    for i, (query_idx, nn_list) in enumerate(nearest_neighbors):
        nearest_neighbor = nn_list[0]
        if nearest_neighbor[2]:
            num_relevant_found += 1
        precision = num_relevant_found / (i + 1)
        recall = num_relevant_found / number_of_positives
        delta_recall = recall - prev_recall
        micro_ap += precision * delta_recall
        prev_recall = recall
        threshold = nearest_neighbor[0]
        precision_recall_history[i] = [precision, recall, threshold]
    
    if save_path is not None:
        print(f'Saving precision-recall history to {save_path}/precision_recall_history.npy...')
        np.save(f'{save_path}/precision_recall_history.npy', precision_recall_history)
        print(f'Saving microAP value to {save_path}/microAP.txt...')
        with open(f'{save_path}/microAP.txt', 'w') as f:
            f.write(str(micro_ap))

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

def _process_gt_dict(
        ground_truth: Dict[int, int],
        n_reference_paths: int,
        n_queries: int
        ):
    ref_grouping = [{
        "query_indices": [],
        "reference_indices": []
    } for _ in range(n_reference_paths)]
    
    for query_idx, ref_idx in ground_truth.items():
        if query_idx >= REFERENCE_SET_SIZE + n_queries:
            break
        ref_set = ref_idx // REFERENCE_SET_SIZE
        ref_grouping[ref_set]["query_indices"].append(query_idx)
        ref_grouping[ref_set]["reference_indices"].append(ref_idx)
    
    return ref_grouping

def _retrieve_array_indices(
        type: Literal["query", "reference"],
        group_matches: Dict[str, List],
        indices: np.array):
    group_indices = group_matches[f"{type}_indices"]
    array_indices = []
    for idx in group_indices:
        array_indices.append(np.where(indices == idx)[0][0])
    
    return array_indices

def compute_gt_similarities(
        query_embeddings_path: str,
        reference_embeddings_paths: List[str],
        model: Literal['clip', 'sae'],
        ground_truth: Dict[int, int],
        n_queries: int,
        save_path: str | None = None,
        top_k: int | None = None 
):
    if len(reference_embeddings_paths) != 20:
        raise ValueError(f"Expected 20 refernece paths. Got {len(reference_embeddings_paths)}.")
    
    ref_grouping = _process_gt_dict(ground_truth, len(reference_embeddings_paths), n_queries)
    model_embeddings_filename = model + '.h5'
    indices_filename = 'indicies.h5' # there was a typo while saving files - using the same name here
    metric = "manhattan" if model == "sae" else "cosine"

    result = {query_idx: {"gt_ref_idx": ref_idx, "similarity": None} for query_idx, ref_idx in ground_truth.items() if query_idx < REFERENCE_SET_SIZE + n_queries}

    with h5py.File(f"{query_embeddings_path}/{model_embeddings_filename}") as query_embeddings_file, \
         h5py.File(f"{query_embeddings_path}/{indices_filename}") as query_indices_file:
        query_embeddings = query_embeddings_file[f"{model}_embeddings"][:]
        query_indices = query_indices_file["image_indices"][:]
        
        for group_matches, reference_embeddings_path in zip(ref_grouping, reference_embeddings_paths):
            print(f"Processing queries with references from {reference_embeddings_path}...")
            with h5py.File(f"{reference_embeddings_path}/{model_embeddings_filename}") as reference_embeddings_file, \
                 h5py.File(f"{reference_embeddings_path}/{indices_filename}") as reference_indices_file:
                reference_embeddings = reference_embeddings_file[f"{model}_embeddings"][:]
                reference_indices = reference_indices_file["image_indices"][:]

                group_query_array_indices = _retrieve_array_indices("query", group_matches, query_indices)
                group_reference_array_indices = _retrieve_array_indices("reference", group_matches, reference_indices)
                group_query_embeddings = query_embeddings[group_query_array_indices, :]
                group_reference_embeddings = reference_embeddings[group_reference_array_indices, :]
                
                if model == "sae" and top_k is not None:
                    _filter_top_k_activations(group_query_embeddings, top_k)
                    _filter_top_k_activations(group_reference_embeddings, top_k)
                
                similarites = compute_similarity(group_query_embeddings, group_reference_embeddings, metric)

                for i, query_idx in enumerate(group_matches["query_indices"]):
                    result[query_idx]["similarity"] = similarites[i, i]
    
    if save_path is not None:
        if top_k is not None:
            filename = f"gt_similarities_{n_queries}_queries_{model}_{top_k}.pkl"
        else:
            filename = f"gt_similarities_{n_queries}_queries_{model}.pkl"
        print(f"Saving gt similarities to {filename}...")
        with open(f"{save_path}/{filename}", 'wb') as f:
            pickle.dump(result, f)

    return result

"""def compute_metrics(
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
    
    return result"""