import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from clip.clip import load
import os
import glob

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

st.set_page_config(layout="wide")
st.title("CLIP vs SAE Image Similarity Explorer")

@st.cache_resource
def load_models():
    try:
        concept_names = load_concepts("../concept_names/clip_ViT-B_16_concept_names.csv")
        ViT_B_16_clip, image_transform = load("ViT-B/16")
        sparse_autoencoder = load_sae("../sae_checkpoints/clip_ViT-B_16_sparse_autoencoder_final.pt")
        return concept_names, ViT_B_16_clip, image_transform, sparse_autoencoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None
    
with st.spinner("Loading models..."):
    concept_names, ViT_B_16_clip, image_transform, sparse_autoencoder = load_models()

if 'samples' not in st.session_state:
    st.session_state.samples = None
    st.session_state.sae_activations = None

data_source = st.radio("Select image source:", 
                      ["Use example dataset", "Upload your own images"], 
                      index=0)

if data_source == "Use example dataset":
    if st.button("Load Dataset"):
        with st.spinner("Loading dataset..."):
            try:
                folder_path = "Rapidata_Other_Animals_10_sample"
                image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))

                samples = []
                for img_path in image_paths:
                    img = Image.open(img_path).convert("RGB")
                    samples.append({"image": img})

                st.session_state.samples = samples
                st.write(f'session state samples: {len(st.session_state.samples)}')

                                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                st.warning("Using fallback mode with limited functionality. You can upload your own images instead.")
                
                
elif data_source == "Upload your own images":
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        images = []
        for file in uploaded_files:
            file.seek(0)
            img = Image.open(file).convert("RGB")
            images.append({"image": img})
        st.session_state.samples = images
        st.success(f"{len(uploaded_files)} images loaded successfully!")


if st.session_state.samples is None or len(st.session_state.samples) == 0:
    st.info("Please load a dataset or upload images to continue")
    st.stop()

max_index = len(st.session_state.samples) - 1
query_idx = st.number_input("Select image index", min_value=0, max_value=max_index, value=min(69, max_index), step=1)
n_concepts = st.slider("Number of concepts to display", min_value=1, max_value=20, value=10, step=1)

if st.button("Analyze Concepts"):
    with st.spinner("Extracting concepts..."):
        try:
            image = st.session_state.samples[query_idx]["image"]
            
            image, image_transformed = prepare_image_from_datasets(image, image_transform)

            top_n_concept_activations, top_n_concept_names = extract_concepts(
                n_concepts=n_concepts,
                concept_names=concept_names,
                image_transformed=image_transformed,
                clip_model=ViT_B_16_clip,
                sae=sparse_autoencoder,
            )

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].imshow(image)
            axs[0].axis("off")
            axs[0].set_title(f"Selected Image (Index {query_idx})")
            
            axs[1].barh(top_n_concept_names, top_n_concept_activations, color="#4e79a7")
            axs[1].set_xlabel("Activation Value")
            axs[1].set_ylabel("Concept")
            axs[1].set_title(f"Top {n_concepts} Concepts")
            
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Extracted Concepts")
            concepts_with_scores = list(zip(top_n_concept_names, top_n_concept_activations))
            concepts_with_scores.sort(key=lambda x: x[1], reverse=True)

            sorted_names = [c[0] for c in concepts_with_scores]
            sorted_scores = [c[1] for c in concepts_with_scores]
            st.table({
                "Concept": sorted_names,
                "Activation Score": [f"{x:.4f}" for x in sorted_scores]
            })
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")