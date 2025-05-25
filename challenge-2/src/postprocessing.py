"""
Author: Annam.ai IIT Ropar
Team Name: SoilClassifiers
Team Members: Siddhant Bhardwaj, Sivadhanushya 
Leaderboard Rank: 36 
"""

import pandas as pd

def format_predictions_for_submission(image_ids, predicted_labels, submission_filename):
    if len(image_ids) != len(predicted_labels):
        print("Postprocessing Error: Mismatch between the number of image IDs and predicted labels.")
        return False
    
    try:
        submission_df = pd.DataFrame({
            'image_id': image_ids,
            'label': predicted_labels
        })
        submission_df.to_csv(submission_filename, index=False)
        print(f"Postprocessing: Submission file '{submission_filename}' created successfully.")
        print(f"  Number of entries: {len(submission_df)}")
        if not submission_df.empty:
            print("  Predicted label distribution:")
            print(submission_df['label'].value_counts(normalize=True).to_string())
        return True
    except Exception as e:
        print(f"Postprocessing Error: Failed to create submission file. Details: {e}")
        return False

def main_postprocessing_workflow_summary():
    print("\n--- Post-processing Workflow Summary ---")
    print("1. Label Conversion: Raw outputs converted into binary labels")
    print("2. CSV Formatting: Creates DataFrame with image_id and label columns")
    print("3. File Saving: Saves DataFrame as required CSV format")
    print("4. No complex post-processing techniques applied")
    print("------------------------------------")
    return 0

if __name__ == '__main__':
    print("--- Running Postprocessing File ---")
    main_postprocessing_workflow_summary()
    print("--- Postprocessing File Finished ---")
