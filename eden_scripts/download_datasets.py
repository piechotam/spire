import fsspec
import zipfile
import requests
import os
import h5py
import argparse

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm
from typing import Union
from PIL import Image

from clip.clip import load as load_clip
from concept_extraction.concept_extraction import load_sae

URLS = {
    "final": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/final_queries.zip",
    "ref1": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_0.zip",
    "ref2": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_1.zip",
    "ref3": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_2.zip",
    "ref4": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_3.zip",
    "ref5": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_4.zip",
    "ref6": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_5.zip",
    "ref7": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_6.zip",
    "ref8": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_7.zip",
    "ref9": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_8.zip",
    "ref10": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_9.zip",
    "ref11": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_10.zip",
    "ref12": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_11.zip",
    "ref13": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_12.zip",
    "ref14": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_13.zip",
    "ref15": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_14.zip",
    "ref16": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_15.zip",
    "ref17": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_16.zip",
    "ref18": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_17.zip",
    "ref19": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_18.zip",
    "ref20": "https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_19.zip",
}

def download(fs, url: str, destination_path: str, chunk_size: int) -> None:
    """
    Downloads the zip from url.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with fs.open(destination_path, "wb") as f:
        with tqdm(total=total_size,
                  unit='B',
                  unit_scale=True,
                  unit_divisor=1024) as bar:
                for chunk in response.iter_content(chunk_size):
                    f.write(chunk)
                    bar.update(len(chunk))

def extract(fs, extract_path: str, zip_path: str):
    """
    Extracts the zip.
    """
    fs.makedirs(extract_path, exist_ok=True)
    with fs.open(zip_path, "rb") as f, zipfile.ZipFile(f, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Removing zip file...")
    fs.rm(zip_path)

def load_models(checkpoint_path: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Loads clip and sae models. 
    """
    print(f"Using device: {device}")
    clip, preprocess = load_clip("ViT-B/16", device)
    clip = clip.to(device)
    sae = load_sae(checkpoint_path, device)
    sae = sae.to(device)

    return clip.eval(), sae.eval(), preprocess

def get_embeddings(fs,
                   images_path: str,
                   clip: torch.nn.Module,
                   clip_embedding_dim: int,
                   sae: torch.nn.Module,
                   sae_embedding_dim: int,
                   preprocess: torchvision.transforms.transforms.Compose,
                   batch_size: int,
                   num_workers: int,
                   output_path: str):
    """
    Gets embeddings for every image and saves them to a file.
    """
    dataset = ImageDataset(images_path, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    fs.makedirs(output_path, exist_ok=True)
    clip_output_path = os.path.join(output_path, "clip.h5")
    sae_output_path = os.path.join(output_path, "sae.h5")
    indices_output_path = os.path.join(output_path, "indicies.h5")

    # Determine the device from the CLIP model
    device = next(clip.parameters()).device

    with h5py.File(clip_output_path, "w") as clip_file, \
         h5py.File(sae_output_path, "w") as sae_file, \
         h5py.File(indices_output_path, "w") as indices_file:
        clip_dataset = clip_file.create_dataset("clip_embeddings",
                                                 shape=(0, clip_embedding_dim),
                                                 maxshape=(None, clip_embedding_dim),
                                                 dtype=np.float32,
                                                 chunks=True)

        sae_dataset = sae_file.create_dataset("sae_embeddings",
                                               shape=(0, sae_embedding_dim),
                                               maxshape=(None, sae_embedding_dim),
                                               dtype=np.float32,
                                               chunks=True)

        indices_dataset = indices_file.create_dataset("image_indices",
                                                      shape=(0,),
                                                      maxshape=(None,),
                                                      dtype=np.int64,
                                                      chunks=True)

        current_idx_in_h5 = 0
        for image_batch, idxs_batch in tqdm(loader):
            # Move image_batch to the same device as the model
            image_batch = image_batch.to(device)
            with torch.no_grad():
                clip_embeddings = clip.encode_image(image_batch)
                # Assuming sae is on the same device, if not, clip_embeddings might need to be moved
                # or sae input handling checked. Given load_models puts them on same device, this should be fine.
                sae_embeddings, _ = sae(clip_embeddings)

            new_size = current_idx_in_h5 + clip_embeddings.shape[0]
            clip_dataset.resize(new_size, axis=0)
            sae_dataset.resize(new_size, axis=0)
            indices_dataset.resize(new_size, axis=0)

            # Move embeddings to CPU before saving with h5py if they are on CUDA
            clip_dataset[current_idx_in_h5:new_size] = clip_embeddings.cpu().numpy()
            sae_dataset[current_idx_in_h5:new_size] = sae_embeddings.cpu().numpy()
            indices_dataset[current_idx_in_h5:new_size] = idxs_batch.cpu().numpy() # idxs_batch is likely already on CPU
            current_idx_in_h5 = new_size

class ImageDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_names = os.listdir(img_dir)
        self.n_images = len(self.image_names)
    def __len__(self):
        return self.n_images
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        dataset_index = int(img_name.split(".")[0][1:])
        return image, dataset_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to download. Default: final.", type=str, default="final")
    parser.add_argument("--num_workers", help="num_workers for DataLoader.", type=int, default=0)
    parser.add_argument("--chunk_size", help="chunk_size for download. Default: 10MB.", type=int, default=(10 * 1024 * 1024))
    parser.add_argument("--data_path", help="Path of directory to data", type=str, default="data")
    parser.add_argument("--batch_size", help="Batch size for data loader.", type=int, default=1024)
    args = parser.parse_args()

    fs = fsspec.filesystem("local")
    fs.makedirs(args.data_path, exist_ok=True)
    
    print("Loading models...")
    clip, sae, preprocess = load_models("sae_checkpoints/clip_ViT-B_16_sparse_autoencoder_final.pt")
    CLIP_EMBEDDING_DIM = 512
    SAE_EMBEDDING_DIM = 4096
    used_names = dict()

    url = URLS.get(args.dataset, None)
    if url is None:
        raise ValueError(f"Dataset {args.dataset} not found. Available datasets: {list(URLS.keys())}")

    extract_path = os.path.join(args.data_path, args.dataset)
    if fs.exists(extract_path):
        print(f"Directory {extract_path} already exists. Skipping extraction.")
    else:
        destination_path = os.path.join(args.data_path, f"{args.dataset}.zip")
        if fs.exists(destination_path):
            print(f"File {destination_path} already exists. Skipping download.")
        else:
            print(f"Downloading zip file from {url} to {destination_path}...")
            download(fs, url, destination_path, args.chunk_size)
        
        print(f"Extracting files to {extract_path}...")
        extract(fs, extract_path, destination_path)
    
    save_path = os.path.join(args.data_path, f"{args.dataset}_embeddings")
    if fs.exists(save_path):
        print(f"Directory {save_path} already exists. Skipping embedding calculation.")
    else:
        fs.makedirs(save_path, exist_ok=True)
        sub_dir_name = "final_queries" if args.dataset == "final" else "references"
        image_path = os.path.join(extract_path, "images", sub_dir_name)
        print(f"Calculating embeddings from {image_path} to {save_path}...")
        
        get_embeddings(fs=fs,
                       images_path=image_path,
                       clip=clip,
                       clip_embedding_dim=CLIP_EMBEDDING_DIM,
                       sae=sae,
                       sae_embedding_dim=SAE_EMBEDDING_DIM, 
                       preprocess=preprocess, 
                       batch_size=args.batch_size, 
                       num_workers=args.num_workers,
                       output_path=save_path)

    print("Done!")