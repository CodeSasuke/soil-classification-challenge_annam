import pandas as pd

# Postprocessing module for soil classification
def build_submission(predictions, image_ids, label_map, output_path):
    """
    predictions: list of integer labels
    image_ids: list of image_id strings
    label_map: dict mapping int->soil_type string
    output_path: where to save CSV
    """
    labels = [label_map[p] for p in predictions]
    submission = pd.DataFrame({
        'image_id': image_ids,
        'soil_type': labels
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
