# AI-Driven Basketball Performance Analytics

This project uses a PyTorch LSTM model to predict basketball shot outcomes and suggest optimal shooting trajectories for missed shots, leveraging ball trajectory and static features.

## Instructions to Use

### Prerequisites
- Python 3.8+
- Install dependencies: `pip install torch pandas numpy sklearn matplotlib seaborn`

### Files Description

1. **train.py**
   - **Description**: Trains the `ShotPredictor` model on the dataset and saves the model weights and scalers.
   - **Input**: Training, validation, datasets at `/content/drive/MyDrive/smai_project/dataset/dataset15/`
   - **Output**: Model weights (`shot_model.pth`), scalers (`scaler_ball.pkl`, `scaler_static.pkl`) at `/content/drive/MyDrive/smai_project/model/model15/`, and training plots at `/content/drive/MyDrive/smai_project/output_graphs/`

2. **eval.py**
   - **Description**: Evaluates the trained model on the test dataset, computing metrics like accuracy, precision, recall, F1-score, confusion matrix, and AUC-ROC.
   - **Input**: Test dataset at `/content/drive/MyDrive/smai_project/dataset/dataset15/` and model weights at `/content/drive/MyDrive/smai_project/model/model15/`
   - **Output**: Evaluation metrics in console and plots (`test_metrics.png`, `confusion_matrix.png`, `roc_curve.png`) at `/content/drive/MyDrive/smai_project/output_graphs/`

3. **infer.py**
   - **Description**: Performs inference on up to 10 missed shots, suggesting optimal trajectories, and visualizes feedback.
   - **Input**: Test dataset at `/content/drive/MyDrive/smai_project/dataset/dataset15/` and model weights at `/content/drive/MyDrive/smai_project/model/model15/`
   - **Output**: Inference logs in console and visualization plots (`shot_{shot_id}_feedback.png`) at `/content/drive/MyDrive/smai_project/output_graphs/`

### Dataset and Model Weights
- **Dataset**: Access the dataset [here]([https://drive.google.com/drive/folders/your-dataset-folder-id](https://drive.google.com/drive/folders/1hPWqSlkWbCqs7E1m6KvizrVXvLOix-KA?usp=drive_link)).
- **Model Weights**: Download the trained model weights [here]([https://drive.google.com/drive/folders/your-model-folder-id](https://drive.google.com/drive/folders/13GeHyJfwCHUMA-CUH_1ioNbgEkOW_nYg?usp=drive_link)).

### Usage
1. Clone this repository.
2. Place the dataset and model weights in the specified directories as per the file descriptions.
3. Run the scripts in order: `python train.py`, then `python eval.py`, and finally `python infer.py`.
