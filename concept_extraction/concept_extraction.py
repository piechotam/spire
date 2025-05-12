import csv
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sparse_autoencoder.autoencoder.model import SparseAutoencoder

_SAE_N_INPUT_FEATURES = 512
_SAE_CONCEPTS_SCALING = 8


def load_concepts(file_path: str, column_index: int = 1) -> tuple:
    """
    Loads a specific column from a CSV file into a Python tuple efficiently.

    Parameters:
    ----------
    file_path: str
        The path to the CSV file.

    column_index: int
        The zero-based index of the column to load.

    Returns:
    --------
    column_data: tuple
        A tuple containing the data from the specified column.
        Returns an empty tuple if the file is not found or an error occurs.
    """

    try:
        with open(file_path, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            column_data = tuple(row[column_index] for row in reader if row)
        return column_data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return ()
    except IndexError:
        print(f"Error: Column index '{column_index}' is out of bounds.")
        return ()
    except Exception as e:
        print(f"An error occurred: {e}")
        return ()


def load_sae(
    checkpoint_path: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.nn.Module:
    """
    Load a Sparse Autoencoder (SAE) model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path: str
        The path to the checkpoint file.

    device: Union[str, torch.device], optional
        The device to load the model on. Defaults to "cuda" if available, else "cpu".

    Returns
    -------
    sparse_autoencoder: torch.nn.Module
        The loaded Sparse Autoencoder model.
        Returns None if the checkpoint file is not found or an error occurs.
    """
    try:
        sae_state_dict = torch.load(
            checkpoint_path, map_location=torch.device(device), weights_only=True
        )

        for param_name, param_tensor in sae_state_dict.items():
            sae_state_dict[param_name] = param_tensor.squeeze()

        sparse_autoencoder = SparseAutoencoder(
            n_input_features=_SAE_N_INPUT_FEATURES,
            n_learned_features=_SAE_N_INPUT_FEATURES * _SAE_CONCEPTS_SCALING,
        )
        sparse_autoencoder.load_state_dict(sae_state_dict)
        return sparse_autoencoder
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at '{checkpoint_path}'")
        return None
    except RuntimeError as e:
        print(f"Error loading checkpoint '{checkpoint_path}': {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def prepare_image(
    image_path: str, preprocess: torchvision.transforms.transforms.Compose
):
    """
    Load and transform image.

    Parameters
    ----------
    image_path: str
        The path to the image.

    preprocess: torchvision.transforms.transforms.Compose
        A torchvision transform that converts a PIL image into a tensor.

    Returns
    -------
    Tuple with loaded PIL.Image and transformed image as tensor.
    """
    image = Image.open(image_path)
    transformed_image = preprocess(image).unsqueeze(0)

    return image, transformed_image


def prepare_image_from_datasets(
    image: Image, preprocess: torchvision.transforms.transforms.Compose
):
    image = image.convert("RGB")

    transformed_image = preprocess(image).unsqueeze(0)
    return image, transformed_image


def extract_concepts(
    n_concepts: int,
    concept_names: tuple,
    image_transformed: torch.Tensor,
    clip_model: torch.nn.Module,
    sae: torch.nn.Module,
):
    """
    Extract top concepts from an image.

    Parameters
    ----------
    n_concepts: int
        Number of concepts to extract.

    concept_names: tuple
        Tuple of named concepts.

    image_transformed: torch.Tensor
        Tensor with transformed image.

    clip_model: torch.nn.Module
        Clip model for creating image embedding.

    sae: torch.nn.Module
        Sparse autoencoder for extracting concepts from clip embedding.

    Returns
    -------
    top_n_concept_activations: np.ndarray
        Activations values for top n_concepts concepts.

    top_n_concept_names: list
        Names of top n_concepts concepts.
    """
    image_clip_embedding = clip_model.encode_image(image_transformed)
    concept_activations, _ = sae(image_clip_embedding)
    concept_activations_numpy = concept_activations.squeeze().detach().numpy()

    indices_of_largest = np.argpartition(concept_activations_numpy, -n_concepts)[
        -n_concepts:
    ]
    largest_indices_sorted = indices_of_largest[
        np.argsort(concept_activations_numpy[indices_of_largest])
    ]

    top_n_concept_activations = concept_activations_numpy[largest_indices_sorted]
    top_n_concept_names = [concept_names[i] for i in largest_indices_sorted]

    return top_n_concept_activations, top_n_concept_names


def extract_embedding(
    image_transformed: torch.Tensor,
    clip_model: torch.nn.Module,
):
    return clip_model.encode_image(image_transformed)



def find_neighbours(query_embedding, embeddings, query_image_idx, top_n=5):
    if len(query_embedding.shape) > 2:
        query_embedding = query_embedding.reshape(1, -1)
    
    if len(embeddings.shape) > 2:
        original_shape = embeddings.shape
        n_samples = original_shape[0]
        embeddings = embeddings.reshape(n_samples, -1)
    
    cos_sim = cosine_similarity(query_embedding, embeddings)
    cos_sim = cos_sim[0] 
    
    sorted_idx = np.argsort(cos_sim)[::-1]
    
    sorted_idx = sorted_idx[sorted_idx != query_image_idx]
    
    nearest_idx = sorted_idx[:top_n]
    farthest_idx = sorted_idx[-top_n:]
    
    nearest_similarities = np.array([cos_sim[idx] for idx in nearest_idx])
    farthest_similarities = np.array([cos_sim[idx] for idx in farthest_idx])
    
    return nearest_idx, nearest_similarities, farthest_idx, farthest_similarities


def extract_sae_activations(embedding, sparse_autoencoder):
    with torch.no_grad():
        activations = sparse_autoencoder.encoder(embedding)
    return activations

def find_neighbours_sae(query_activations, all_activations, query_idx, top_n=5):
    if len(query_activations.shape) > 2:
        query_activations = query_activations.reshape(1, -1)
    if len(all_activations.shape) > 2:
        original_shape = all_activations.shape
        n_samples = original_shape[0]
        all_activations = all_activations.reshape(n_samples, -1)
    
    
    distances = np.abs(all_activations - query_activations).sum(axis=1)
    n_samples = len(distances)
    if query_idx >= n_samples:
        print(f"Warning: query_idx ({query_idx}) is outside valid range (0-{n_samples-1})")
    else:
        distances[query_idx] = -1
    
        top_n = min(top_n, n_samples - 1)
        
        sorted_idx = np.argsort(distances)
        sorted_idx = sorted_idx[sorted_idx != query_idx]
        nearest_idx = sorted_idx[:top_n]
        farthest_idx = sorted_idx[-top_n:]
        
        nearest_similarities = np.array([distances[idx] for idx in nearest_idx])
        farthest_similarities = np.array([distances[idx] for idx in farthest_idx])
    
    return nearest_idx, nearest_similarities, farthest_idx, farthest_similarities

def visualize_neighbours_with_distances(
    samples,
    query_idx,
    nearest_indices,
    nearest_similarities,
    farthest_indices=None,
    farthest_similarities=None,
    title="Image Similiarity Comparison (CLIP)",
):

    show_farthest = farthest_indices is not None and farthest_similarities is not None

    n_nearest = len(nearest_indices)
    n_farthest = len(farthest_indices) if show_farthest else 0

    if show_farthest:
        n_cols = max(n_nearest, n_farthest) + 1
        n_rows = 2
    else:
        n_cols = n_nearest + 1
        n_rows = 1

    _, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    if n_rows == 1:
        axs = axs.reshape(1, -1)

    axs[0, 0].imshow(samples[query_idx]["image"])
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Query Image")

    for i, (neighbour_idx, similarity) in enumerate(
        zip(nearest_indices, nearest_similarities)
    ):
        axs[0, i + 1].imshow(samples[neighbour_idx]["image"])
        axs[0, i + 1].axis("off")
        axs[0, i + 1].set_title(f"Nearest {i+1}\nSimilarity: {similarity:.4f}")

    for i in range(n_nearest + 1, n_cols):
        axs[0, i].axis("off")
        axs[0, i].set_visible(False)

    if show_farthest:
        axs[1, 0].imshow(samples[query_idx]["image"])
        axs[1, 0].axis("off")
        axs[1, 0].set_title("Query Image")

        for i, (far_idx, similarity) in enumerate(
            zip(farthest_indices, farthest_similarities)
        ):
            axs[1, i + 1].imshow(samples[far_idx]["image"])
            axs[1, i + 1].axis("off")
            axs[1, i + 1].set_title(f"Farthest {i+1}\nSimilarity: {similarity:.4f}")

        for i in range(n_farthest + 1, n_cols):
            axs[1, i].axis("off")
            axs[1, i].set_visible(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.show()