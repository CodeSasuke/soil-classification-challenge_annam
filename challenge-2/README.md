# One-Class Soil Image Classification using OpenAI CLIP

**Author:** Siddhant Bhardwaj  
**Team Name:** Siddhant Bhardwaj
**Team Members:** Siddhant Bhardwaj, Sivadhanushya  
**Leaderboard Rank:** 36 (as of last update)  
**Date:** May 24, 2025

## Project Overview

This project addresses a one-class image classification task: determining whether an input image is "soil" or "not soil". The primary challenge is that the training dataset consists *only* of "soil" images, with no explicit examples of "not soil" images. This necessitates a one-class learning or anomaly detection approach.

The implemented solution leverages the powerful semantic feature extraction capabilities of OpenAI's CLIP model (specifically the `ViT-B/16` variant) to define what constitutes "soil" and then classifies test images based on their similarity to this learned representation.

## Approach: CLIP Image Embeddings with Mean Prototype & Percentile Thresholding

The core strategy revolves around creating a "prototype" for the "soil" class in the semantic feature space learned by CLIP and then using a data-driven threshold to classify new images.

### Key Methodological Steps:

1.  **Feature Extractor - OpenAI CLIP (`ViT-B/16`):**
    *   **Rationale:** Instead of traditional CNN features learned from object recognition datasets (like ImageNet), we use CLIP. CLIP is pre-trained on a massive dataset of 400 million image-text pairs from the internet. This allows its image encoder (a Vision Transformer in this case) to learn rich **semantic embeddings**. These embeddings capture higher-level conceptual meaning rather than just low-level visual patterns (textures, colors, shapes).
    *   **Benefit for "Soil":** "Soil" is a visually diverse concept. CLIP's semantic understanding is hypothesized to be more robust in identifying this core concept across its many variations (different colors, textures, moisture levels, compositions) compared to features purely focused on visual object parts. It's expected to better distinguish "soil" from conceptually different "not soil" images.

2.  **"Training" Phase (Defining Normality - `training.ipynb`):**
    *   **No Fine-tuning of CLIP:** The pre-trained CLIP model is used as a fixed feature extractor; it is not fine-tuned on the soil images.
    *   **Step 1: Extract Soil Embeddings:** All images from the training dataset (which are exclusively "soil" images) are passed through the CLIP `ViT-B/16` image encoder. This produces a high-dimensional embedding vector (512 dimensions for ViT-B/16) for each training image.
    *   **Step 2: Create "Soil Prototype":** A single representative vector for the "soil" class is created by calculating the **mean (average) of all the training soil image embeddings**. This mean vector serves as a "prototype" or a central point representing the "average semantic concept of soil" as learned from the provided training data. This prototype is saved (e.g., `soil_prototype_clip_vit_base_patch16.npy`).
    *   **Step 3: Determine Similarity Threshold:**
        *   The **cosine similarity** of each *training soil image embedding* to this *soil prototype* is calculated. This results in a distribution of similarity scores, indicating how "typical" or "central" each training soil image is relative to the average soil concept.
        *   A decision **threshold** is then chosen based on a specific **percentile** of this distribution. For this project, the **5.5th percentile** was used. This means that any image (training or test) whose embedding is less similar to the prototype than the 5.5% "least typical" (but still valid according to the training data) training soil images will be considered an anomaly.
        *   This data-driven threshold is saved (e.g., `similarity_threshold_clip_vit_base_patch16_p5.5.txt`). A histogram of these training similarities is also typically plotted and saved for analysis.

3.  **Inference Phase (Classifying Test Images - `inference.ipynb`):**
    *   **Step 1: Load Artifacts:** The pre-calculated "soil prototype" and the "similarity threshold" are loaded from the files saved during the "training" phase. The CLIP model is also loaded.
    *   **Step 2: Extract Test Image Embeddings:** Each image from the test dataset is passed through the same CLIP `ViT-B/16` image encoder to obtain its semantic embedding.
    *   **Step 3: Calculate Similarity:** The cosine similarity between each test image's embedding and the loaded "soil prototype" is computed.
    *   **Step 4: Classification:**
        *   If a test image's similarity to the prototype is **greater than or equal to** the loaded similarity threshold, it is classified as "soil" (label 1).
        *   Otherwise, it is classified as "not soil" (label 0).
    *   **Step 5: Generate Submission:** The predictions are formatted into a CSV file with `image_id` and `label` columns as per competition requirements.

### Why this approach was chosen and successful:

*   **Semantic Power:** CLIP's ability to understand images at a conceptual level is key. It allows the model to group varied "soil" images based on their underlying meaning rather than just superficial appearance. This makes it more robust to variations in soil type, lighting, and minor occlusions.
*   **Robust Prototype:** The mean embedding in a rich semantic space provides a stable central point for the "soil" class.
*   **Data-Driven and Tunable Threshold:** Using a percentile of training similarities provides a principled way to set the decision boundary. This percentile (5.5% in the final version) is a hyperparameter that can be tuned (and was, leading to this choice) to optimize performance metrics like F1-score on a leaderboard or validation set.
*   **Effectiveness for One-Class Problems:** This prototype-similarity method is a strong approach for anomaly detection when only positive class examples are available for training. It defines a "normal" region in the feature space based on the known good samples.

## File Structure and Usage:

*   **`requirements.txt`**: Lists all necessary Python libraries to run the code. Install using `pip install -r requirements.txt`.
*   **`preprocessing.py`**:
    *   Contains functions and explanations related to how images are loaded and preprocessed by the `CLIPProcessor` before being fed into the CLIP model.
    *   The CLIP processor handles tasks like resizing images to the model's expected input size (e.g., 224x224 for ViT-B/16) and normalizing pixel values.
    *   No other dataset-wide preprocessing (like global scaling or PCA on raw pixels) is performed before CLIP embedding extraction in this method.
*   **`postprocessing.py`**:
    *   Outlines the minimal post-processing steps. For this project, it mainly involves:
        1.  Converting the binary classification output (from similarity comparison) into labels 0 ('not soil') and 1 ('soil').
        2.  Formatting these predictions into the required two-column CSV (`image_id`, `label`) for submission.
*   **`training.ipynb`**:
    *   A Jupyter Notebook that performs the "training" phase:
        *   Loads training image IDs.
        *   Loads the pre-trained CLIP `ViT-B/16` model.
        *   Extracts CLIP embeddings for all training soil images.
        *   Calculates the mean "soil prototype" embedding and saves it to a `.npy` file.
        *   Calculates cosine similarities of training embeddings to this prototype.
        *   Determines the 5.5th percentile similarity threshold and saves it to a `.txt` file.
        *   Optionally plots the distribution of training similarities.
*   **`inference.ipynb`**:
    *   A Jupyter Notebook that performs the inference on the test set:
        *   Loads the pre-trained CLIP `ViT-B/16` model.
        *   Loads the saved "soil prototype" (`.npy`) and similarity threshold (`.txt`) from the "training" phase.
        *   Loads test image IDs.
        *   Extracts CLIP embeddings for all test images.
        *   Calculates cosine similarities of test embeddings to the soil prototype.
        *   Classifies images based on the loaded threshold.
        *   Generates the final `submission_*.csv` file.

## How to Run:

1.  **Setup Environment:**
    *   Ensure you have a Python environment with GPU support (recommended for CLIP).
    *   Install all libraries listed in `requirements.txt`.
2.  **Dataset:**
    *   Place the competition dataset in the structure expected by `BASE_PATH` (e.g., `/kaggle/input/soil-classification-part-2/soil_competition-2025/` containing `train_labels.csv`, `test_ids.csv`, and `train/`, `test/` image folders).
3.  **Run Training Notebook (`training.ipynb`):**
    *   Execute the cells in this notebook. This will generate the `soil_prototype_*.npy` and `similarity_threshold_*.txt` files in the default working directory (e.g., `/kaggle/working/`).
4.  **Run Inference Notebook (`inference.ipynb`):**
    *   Ensure the paths to the prototype and threshold files generated by `training.ipynb` are correctly specified.
    *   Execute the cells. This will generate the final submission CSV file in the working directory.

## Potential Further Improvements (Beyond this specific implementation):

*   **Experiment with other CLIP Model Variants:** Try larger CLIP models (e.g., `ViT-L/14`) if computational resources allow, which might provide even more discriminative embeddings, though with increased processing time.
*   **Advanced Thresholding:** Explore more sophisticated methods for threshold selection if a validation set with anomalies becomes available.
*   **Prompt Engineering (Alternative CLIP method):** If the prototype method hits a ceiling, exploring direct prompt-based classification (comparing image embeddings to text embeddings of "a photo of soil" vs. "a photo of a car", etc.) could be another avenue, though it requires careful prompt design.
*   **Ensembling:** If other distinct one-class approaches show promise on different types of errors, ensembling could be considered.

This detailed approach using CLIP's semantic embeddings and a data-driven prototype has proven to be highly effective for this one-class soil classification challenge.