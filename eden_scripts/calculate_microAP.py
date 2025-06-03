import fsspec
import argparse

from metrics.metrics import create_ground_truth_dict, create_matched_pairs, micro_average_precision

QUERY_EMBEDDINGS_PATH = "final_embeddings"
REFERENCE_EMBEDDINGS_PATHS = [f"ref{i}_embeddings" for i in range(1, 21)]
NUMBER_OF_POSITIVES = 10000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", help="Path of directory to save the output. Default: results.", type=str, default="results")
    parser.add_argument("--model", help="Which model embeddings to use. Options: clip, sae.", type=str)
    parser.add_argument("--batch_size", help="Size of the batch of queries processed at once.", type=int)
    args = parser.parse_args()

    avail_models = ["clip", "sae"]
    if args.model not in avail_models:
        raise ValueError(f"Invalid model {args.model}. Available models: {avail_models}")

    fs = fsspec.filesystem("local")
    print(f"Creating directory {args.save_path} for saving the output...")
    fs.makedirs(args.save_path, exist_ok=True)

    print(f"Parsing ground truth file...")
    ground_truth = create_ground_truth_dict("gt.csv")

    print(f"Running nearest neighbor search...")
    matched_pairs = create_matched_pairs(QUERY_EMBEDDINGS_PATH, REFERENCE_EMBEDDINGS_PATHS, args.model, ground_truth, args.batch_size, args.save_path)

    print(f"Calculating microAP...")
    micro_ap, _ = micro_average_precision(matched_pairs, NUMBER_OF_POSITIVES, args.save_path)
    print(f"Micro Average Precision: {micro_ap}")
    print("Done!")