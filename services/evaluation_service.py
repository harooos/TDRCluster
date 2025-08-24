import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder

def evaluate_clustering(ground_truth_file: str, prediction_file: str) -> dict:
    """
    Calculates clustering evaluation metrics (MI, NMI, AMI) from two CSV files.

    Args:
        ground_truth_file (str): Path to the CSV file with ground truth labels.
                                 The file should have two columns: 'query' and 'label'.
        prediction_file (str): Path to the CSV file with predicted cluster labels.
                               The file should have two columns: 'query' and 'label'.

    Returns:
        dict: A dictionary containing the calculated scores for MI, NMI, and AMI.
    """
    # Read the CSV files
    try:
        ground_truth_df = pd.read_csv(ground_truth_file, header=None, names=['query', 'true_label'])
        prediction_df = pd.read_csv(prediction_file, header=None, names=['query', 'pred_label'])
    except FileNotFoundError as e:
        return {"error": str(e)}

    # Merge the two dataframes on the 'query' column to align labels
    merged_df = pd.merge(ground_truth_df, prediction_df, on='query')

    if merged_df.empty:
        return {"error": "No common queries found between the two files. Please check the 'query' columns."}

    true_labels = merged_df['true_label']
    pred_labels = merged_df['pred_label']

    # Encode string labels to integers
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_labels)
    pred_labels_encoded = le.fit_transform(pred_labels)

    # Calculate metrics
    mi = mutual_info_score(true_labels_encoded, pred_labels_encoded)
    nmi = normalized_mutual_info_score(true_labels_encoded, pred_labels_encoded)
    ami = adjusted_mutual_info_score(true_labels_encoded, pred_labels_encoded)

    scores = {
        'MI': mi,
        'NMI': nmi,
        'AMI': ami
    }

    return scores

if __name__ == '__main__':
    # Example Usage:
    # Create dummy CSV files for demonstration
    
    # Ground truth data
    # truth_data = {
    #     'query': [f'query_{i}' for i in range(10)],
    #     'label': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C']
    # }
    # truth_df = pd.DataFrame(truth_data)
    # truth_df.to_csv('ground_truth.csv', index=False, header=False)

    # # Prediction data (a slightly imperfect clustering)
    # pred_data = {
    #     'query': [f'query_{i}' for i in range(10)],
    #     'label': ['X', 'X', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z']
    # }
    # pred_df = pd.DataFrame(pred_data)
    # pred_df.to_csv('predictions.csv', index=False, header=False)

    # Evaluate
    evaluation_results = evaluate_clustering('data/raw_data/banking77.csv', 'output/banking77_result.csv')

    print("Clustering Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")

    # Clean up dummy files
    # import os
    # os.remove('ground_truth.csv')
    # os.remove('predictions.csv')
