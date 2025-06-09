import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import torch


APP_ROOT = Path(__file__).parent.resolve()
sys.path.append(str(APP_ROOT.parent))
from clip.clip import load

from concept_extraction.concept_extraction import (
    load_concepts,
    load_sae,
    prepare_image,
    prepare_image_from_datasets,
    extract_concepts,
    extract_embedding,
    find_neighbours,
    visualize_neighbours_with_distances,
    extract_sae_activations,
    find_neighbours_sae,
)

CONCEPT_NAMES_PATH = (
    APP_ROOT.parent / "concept_names" / "clip_ViT-B_16_concept_names.csv"
)
RAPIDATA_OTHER_ANIMALS_FOLDER = APP_ROOT / "Rapidata_Other_Animals_10_sample"
SAE_CHECKPOINT = (
    APP_ROOT.parent / "sae_checkpoints" / "clip_ViT-B_16_sparse_autoencoder_final.pt"
)
STL10_FOLDER = APP_ROOT / "stl10_sample_500"


def visualize_neighbours_with_distances_streamlit(
    samples,
    query_idx,
    nearest_indices,
    nearest_similarities,
    farthest_indices=None,
    farthest_similarities=None,
    title="Image Similarity Comparison (CLIP)",
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

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

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

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    st.pyplot(fig)


st.set_page_config(layout="wide")
st.title("CLIP vs SAE Image Similarity Explorer")


@st.cache_resource
def load_models():
    try:
        concept_names = load_concepts(str(CONCEPT_NAMES_PATH))
        ViT_B_16_clip, image_transform = load("ViT-B/16")
        sparse_autoencoder = load_sae(str(SAE_CHECKPOINT))
        return concept_names, ViT_B_16_clip, image_transform, sparse_autoencoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None


@st.cache_resource
def load_embeddings():
    embeddings_clip = None
    embeddings_sae = None
    samples_stl10 = []
    try:
        image_paths = list(STL10_FOLDER.glob("*.png"))[:200]
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            samples_stl10.append({"image": img})
    except Exception as e:
        st.error(f"Error loading STL10 images: {str(e)}")

    try:
        data_clip = torch.load(APP_ROOT / "stl10_embeddings.pt", map_location="cpu")
        embeddings_clip = data_clip["embeddings"]
    except Exception as e:
        st.error(f"Error loading CLIP embeddings: {str(e)}")

    try:
        data_sae = torch.load(APP_ROOT / "stl10_sae_activations.pt", map_location="cpu")
        embeddings_sae = data_sae["sae_activations"]
    except Exception as e:
        st.error(f"Error loading SAE embeddings: {str(e)}")

    return samples_stl10, embeddings_clip, embeddings_sae


with st.spinner("Loading models..."):
    concept_names, ViT_B_16_clip, image_transform, sparse_autoencoder = load_models()

tabs = st.tabs(["Concepts", "Similarities"])

with tabs[0]:
    st.header("Concepts Extraction")
    data_source_1 = st.radio(
        "Select image source:",
        ["Use example dataset (Rapidata_Other_Animals)", "Upload your own images"],
        key="ds1",
    )

    if data_source_1 == "Use example dataset (Rapidata_Other_Animals)":
        if st.button("Load Dataset", key="load1"):
            with st.spinner("Loading dataset..."):
                try:
                    image_paths = list(RAPIDATA_OTHER_ANIMALS_FOLDER.glob("*.jpg"))
                    samples = []
                    for img_path in image_paths:
                        img = Image.open(img_path).convert("RGB")
                        samples.append({"image": img})
                    st.session_state["samples_tab1"] = samples
                    st.toast(f"Loaded {len(samples)} images!")
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")

    elif data_source_1 == "Upload your own images":
        uploaded_files_1 = st.file_uploader(
            "Upload images",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key="upload1",
        )
        if uploaded_files_1:
            samples = []
            for file in uploaded_files_1:
                img = Image.open(file).convert("RGB")
                samples.append({"image": img})
            st.session_state["samples_tab1"] = samples
            st.toast(f"{len(samples)} images uploaded!")

    samples_1 = st.session_state.get("samples_tab1", None)
    if samples_1 is not None and len(samples_1) > 0:
        st.subheader("Select an Image")

        options = [f"Image {i}" for i in range(len(samples_1))]
        selected_idx = st.selectbox("Choose an image:", options, index=0)

        selected_idx_num = options.index(selected_idx)
        st.image(
            samples_1[selected_idx_num]["image"],
            caption=f"Selected Image {selected_idx_num}",
        )

        n_concepts_1 = st.slider(
            "Number of concepts to display",
            min_value=1,
            max_value=20,
            value=10,
            key="nconcepts1",
        )

        if st.button("Analyze Concepts", key="analyze1"):
            with st.spinner("Extracting concepts..."):
                try:
                    image = samples_1[selected_idx_num]["image"]
                    image, image_transformed = prepare_image_from_datasets(
                        image, image_transform
                    )

                    top_n_concept_activations, top_n_concept_names = extract_concepts(
                        n_concepts=n_concepts_1,
                        concept_names=concept_names,
                        image_transformed=image_transformed,
                        clip_model=ViT_B_16_clip,
                        sae=sparse_autoencoder,
                    )

                    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                    axs[0].imshow(image)
                    axs[0].axis("off")
                    axs[0].set_title(f"Selected Image (Index {selected_idx_num})")

                    axs[1].barh(
                        top_n_concept_names, top_n_concept_activations, color="#4e79a7"
                    )
                    axs[1].set_xlabel("Activation Value")
                    axs[1].set_ylabel("Concept")
                    axs[1].set_title(f"Top {n_concepts_1} Concepts")

                    plt.tight_layout()
                    st.pyplot(fig)

                    st.subheader("Extracted Concepts")
                    concepts_with_scores = list(
                        zip(top_n_concept_names, top_n_concept_activations)
                    )
                    concepts_with_scores.sort(key=lambda x: x[1], reverse=True)

                    sorted_names = [c[0] for c in concepts_with_scores]
                    sorted_scores = [c[1] for c in concepts_with_scores]
                    st.table(
                        {
                            "Concept": sorted_names,
                            "Activation Score": [f"{x:.4f}" for x in sorted_scores],
                        }
                    )
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    else:
        st.info("Please load or upload images to continue.")
with tabs[1]:
    st.header("Image Similarity Explorer")
    st.caption("Using 200 samples from the STL10 dataset")

    if "stl10_data_loaded" not in st.session_state:
        with st.spinner("Loading STL10 dataset and embeddings..."):
            samples_stl10, embeddings_clip, embeddings_sae = load_embeddings()
            st.session_state["samples_tab2"] = samples_stl10
            st.session_state["embeddings_clip"] = embeddings_clip
            st.session_state["embeddings_sae"] = embeddings_sae
            st.session_state["stl10_data_loaded"] = True

    samples_2 = st.session_state.get("samples_tab2", [])
    embeddings_clip = st.session_state.get("embeddings_clip", None)
    embeddings_sae = st.session_state.get("embeddings_sae", None)

    embedding_type = st.radio("Select model:", ["CLIP", "SAE"], key="embed_type")
    embeddings = embeddings_clip if embedding_type == "CLIP" else embeddings_sae

    if embeddings is not None and len(samples_2) > 0:
        image_options = [f"Image {i}" for i in range(len(samples_2))]
        selected_option = st.selectbox("Choose an image:", image_options)
        query_idx_2 = image_options.index(selected_option)

        st.image(
            samples_2[query_idx_2]["image"],
            caption=f"Selected Image (Index {query_idx_2})",
            use_container_width=False,
        )

        top_k = st.slider(
            "Number of nearest/farthest images", min_value=1, max_value=10, value=5
        )

        query_embedding = embeddings[query_idx_2].unsqueeze(0)

        if embedding_type == "SAE":
            nearest_idx, nearest_similarities, farthest_idx, farthest_similarities = (
                find_neighbours_sae(
                    query_embedding.detach().numpy(),
                    embeddings.detach().numpy(),
                    query_idx_2,
                    top_n=top_k,
                )
            )
            title = "Image Similarity Comparison (SAE)"
        else:
            nearest_idx, nearest_similarities, farthest_idx, farthest_similarities = (
                find_neighbours(
                    query_embedding.detach().numpy(),
                    embeddings.detach().numpy(),
                    query_idx_2,
                    top_n=top_k,
                )
            )
            title = "Image Similarity Comparison (CLIP)"

        visualize_neighbours_with_distances_streamlit(
            samples_2,
            query_idx=query_idx_2,
            nearest_indices=nearest_idx,
            nearest_similarities=nearest_similarities,
            farthest_indices=farthest_idx,
            farthest_similarities=farthest_similarities,
            title=title,
        )
    else:
        st.info("Loading data or no images available yet.")
