import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# from sklearn.metrics import roc_curve, auc
import pandas as pd


def plot_precision_recall_curve(
    precision_recall_history: np.ndarray, model_name: str
) -> None:
    """
    Plots the Precision-Recall curve for a given model.

    Parameters:
    ----------
    precision_recall_history (np.ndarray): Numpy array where the first column
                                           contains precision values and the second
                                           column contains recall values.
    model_name (str): The name of the model to be displayed in the plot legend and title.

    Returns:
    ----------
    None: This function does not return any value. It displays the plot directly.
    """

    plt.figure(figsize=(8, 6))
    plt.plot(
        precision_recall_history[:, 1],
        precision_recall_history[:, 0],
        label=f"{model_name} PR curve",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({model_name})")
    plt.grid(True)
    plt.legend()
    plt.show()


def similarity_boxplot(matched_pairs: dict, model_name: str) -> None:
    """
    Plots a boxplot comparing the similarity scores of ground truth (GT) matches
    and mismatches from the provided matched pairs.

    Parameters:
    ----------
    matched_pairs (dict): A dictionary where keys are query IDs and values are
                          dictionaries containing similarity scores and a boolean
                          indicating if the match is correct.
                          Example: {qid: {"similarity": float, "correct_match": bool}}
    model_name (str): The name of the model used for generating the similarity scores.

    Returns:
    ----------
    None: This function does not return any value. It displays a boxplot.
    """
    gt_similarities = []
    mismatch_similarities = []

    for qid, match_info in matched_pairs.items():
        sim = match_info["similarity"]
        if match_info["correct_match"]:
            gt_similarities.append(sim)
        else:
            mismatch_similarities.append(sim)
    data = {
        "similarity": gt_similarities + mismatch_similarities,
        "type": ["GT"] * len(gt_similarities)
        + ["Mismatch"] * len(mismatch_similarities),
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="type",
        y="similarity",
        hue="type",
        data=df,
        palette={"GT": "green", "Mismatch": "red"},
        boxprops={"alpha": 0.8},
    )

    plt.title(f"{model_name} - Similarity Score (GT vs Mismatch)")
    plt.xlabel("")
    plt.grid(True)
    plt.show()


def similarity_histogram(knn_dict: dict, model_name: str) -> None:
    """
    Plots a histogram of similarity scores for correct and incorrect matches.

    Parameters:
    ----------
    knn_dict (dict): A dictionary where keys are identifiers and values are lists of tuples. Each tuple contains:
                     - similarity score (float)
                     - data point (any type)
                     - is_correct (bool): True if the match is correct, False otherwise.
    model_name (str): The name of the model being evaluated, used for the plot title.

    Returns:
    ----------
    None: This function does not return any value. It displays a histogram plot.
    """
    correct, incorrect = [], []
    for neighbors in knn_dict.values():
        for sim, _, is_correct in neighbors:
            (correct if is_correct else incorrect).append(sim)

    plt.figure(figsize=(10, 6))
    sns.histplot(correct, color="green", label="Ground Truth", kde=True, stat="density")
    sns.histplot(
        incorrect, color="red", label="Incorrect Match", kde=True, stat="density"
    )
    plt.title(f"{model_name} - Similarity Score Distribution")
    plt.xlabel("Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


# def plot_roc_curve(knn_dict, model_name):
#     y_true, y_scores = [], []
#     for neighbors in knn_dict.values():
#         for sim, _, is_correct in neighbors:
#             y_true.append(1 if is_correct else 0)
#             y_scores.append(sim)

#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)

#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
#     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
#     plt.title(f"ROC Curve ({model_name})")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def similarity_histogram_matched_pairs(matched_pairs: dict, model_name: str) -> None:
    """
    Plots a histogram of similarity scores for matched pairs, distinguishing between ground truth (GT)
    and mismatched pairs.

    Parameters:
    ----------
    matched_pairs (dict): A dictionary where each value is a dictionary containing 'similarity'
                          and 'correct_match' keys. 'similarity' is a float representing the
                          similarity score, and 'correct_match' is a boolean indicating if the
                          pair is a correct match.
    model_name (str): The name of the model being evaluated, used for the plot title.

    Returns:
    ----------
    None: This function does not return a value. It displays a histogram plot.
    """

    gt_similarities = [
        v["similarity"] for v in matched_pairs.values() if v["correct_match"]
    ]
    mismatch_similarities = [
        v["similarity"] for v in matched_pairs.values() if not v["correct_match"]
    ]

    plt.figure(figsize=(10, 6))
    sns.histplot(
        gt_similarities, color="green", label="GT", kde=True, stat="density", bins=30
    )
    sns.histplot(
        mismatch_similarities,
        color="red",
        label="Mismatch",
        kde=True,
        stat="density",
        bins=30,
    )
    plt.xlabel("Similarity")
    plt.ylabel("Density")
    plt.title(f"{model_name} - Similarity Distribution (GT vs Mismatch)")
    plt.legend()
    plt.grid(True)
    plt.show()
