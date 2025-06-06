import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[gt_similarities, mismatch_similarities])
    plt.xticks([0, 1], ["GT", "Mismatch"])
    plt.title(f"{model_name} - Similarity Score (GT vs Mismatch)")
    plt.show()
