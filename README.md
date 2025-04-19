# DA6401-Assignment-2
This assignment implements a Convolutional Neural Network (CNN) for classifying images from the iNaturalist dataset with hyperparameter optimization using Weights & Biases (wandb).

### Overview
The notebook **`cnn_training.ipynb`** trains a CNN model to classify images from the iNaturalist dataset, which contains natural wildlife images across 10 different classes. It implements a custom CNN architecture with configurable parameters and uses Weights & Biases (wandb) for experiment tracking and hyperparameter optimization.

### Project Structure
```
DA6401-Assignment-2-PartA/
    ├── .gitignore              # Includes dataset and wandb folder
    ├── best_model.pth          # Model with the best val accuracy achieved with wandb sweep
    ├── cnn_training.ipynb      # Jupyter notebook - contains all of the code necessary for Part A of the assignment
    ├── README.md               # Project documentation
    ├── test_prediction.png     # A 10x3 grid of sample images from the test set along with their predictions and true labels
    └── val/
```

### Requirements
- Python 3.x
- PyTorch
- torchvision
- NumPy
- scikit-learn
- matplotlib
- PIL (Python Imaging Library)
- Weights & Biases (wandb)

## Setup Instructions
- Install required packages:
    ```
    pip install torch torchvision numpy scikit-learn matplotlib pillow wandb
    ```
    If your device has an nvidia gpu supporting CUDA, install the required drivers, CUDA toolkit and Microsoft Visual Studio 2022. Then install the pytorch packages with:
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```
    **Note:** Ensure that the CUDA toolkit installed is supported by PyTorch. The above installation is compatible with CUDA version 12.6.
- Download the iNaturalist dataset and organize it into the following structure:
    ```
    nature_12K/
    └── inaturalist_12K/
        ├── train/
        │   ├── class_1/
        │   ├── class_2/
        │   └── ...
        └── val/
            ├── class_1/
            ├── class_2/
            └── ...
    ```
- Go to your Weights & Biases account (https://wandb.ai) and authenticate:
    ```
    import wandb
    wandb.login()
    ```

## Code Structure
The notebook is organized into the following sections:
- Model definition (`CNN` class)
- Dataset handling (`iNaturalistDataset` class)
- Training and validation functions
- Hyperparameter optimization with wandb
- Model evaluation

### CNN Model Architecture
The CNN class defines a flexible convolutional neural network with configurable:
- Number of convolutional layers (fixed at 5)
- Filter sizes
- Kernel sizes
- Activation functions
- Dense layer neurons
- Batch normalization
- Dropout rate

```
class CNN(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=10, 
                 filter_sizes=[64, 64, 64, 64, 64], 
                 kernel_sizes=[3, 3, 3, 3, 3],
                 activation_fn=nn.ReLU(),
                 dense_neurons=256,
                 input_size=(224, 224),
                 use_batchnorm=False,
                 dropout_rate=0.0):
```

### Dataset Handling
The iNaturalistDataset class loads and preprocesses images from the dataset:
```
class iNaturalistDataset(Dataset):
    def __init__(self, root_dir, transform=None):
```

This class:
- Identifies class directories
- Loads image paths
- Creates class-to-index mapping
- Applies specified transformations

### Data Splitting Functions
The notebook implements stratified train-validation splits to ensure class distribution is maintained for equal representation in training:
```
def create_stratified_splits(dataset, val_ratio=0.2, random_state=42):
```

### Training Functions
The following functions handle the training process:
- `train_epoch`: Trains the model for one epoch
- `validate`: Evaluates the model on the validation set
- `test`: Evaluates the model on the test set
- `train_model`: Complete training pipeline without wandb
- `train_with_wandb`: Training pipeline with wandb integration

### Running the Code
- To train the model with default parameters, run:  
    ```
    model, val_acc = train_model()
    ```
- To run hyperparameter optimization with wandb, run:
    ```
    sweep_id = wandb.sweep(sweep_config, project='DA6401-Assignment-2')
    wandb.agent(sweep_id, train_with_wandb, count=25)  # Run 25 experiments
    ```
- To evaluate the best model on the test set, run:  
    ```
    evaluate_best_model()
    ```

#### Hyperparameter Optimization
The notebook uses wandb's Bayesian optimization to search for optimal hyperparameters:
```
pythonsweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        # (hyperparameter ranges)
    }
}
```

Optimized hyperparameters include:
- Learning rate: 0.0001 to 0.01
- Batch size: 32, 64, or 128
- Filter organization: same, double, or half
- Kernel size: 3 or 5
- Activation function: ReLU, GELU, SiLU, or Mish
- Batch normalization: True or False
- Dropout rate: 0.0, 0.2, 0.3, or 0.5
- Weight decay: 0.0, 0.0001, or 0.001

#### Evaluation
The evaluation function:
- Loads the best model from saved weights
- Runs inference on the test set
- Calculates accuracy
- Visualizes predictions with a grid of images (saved as 'test_predictions.png')
- Logs results to wandb
The visualization shows predicted vs. true class labels for sample images, with correct predictions highlighted in green and incorrect in red.

## Important Functions
- `CNN.forward()`: The forward pass of the CNN model
    - Passes input through convolutional layers with optional batch normalization
    - Applies activation function after each layer
    - Uses max pooling to reduce spatial dimensions
    - Flattens the output and passes through fully connected layers
    - Returns class logits
- `train_epoch()`: Trains the model for one epoch
    - Sets model to training mode
    - Processes batches of data
    - Computes loss and performs backpropagation
    - Updates model parameters
    - Tracks and returns training loss and accuracy
- `validate()`: Evaluates the model on validation data
    - Sets model to evaluation mode
    - Processes validation data without gradient calculation
    - Computes loss and predictions
    - Returns validation loss and accuracy
- `train_with_wandb()`: Trains with wandb integration
    - Initializes a wandb run with given configuration
    - Sets up data loaders and model
    - Trains the model for specified epochs
    - Logs metrics to wandb after each epoch
    - Saves the best model based on validation accuracy (as 'best_model.pth')
- `test()`: Tests the model performance
    - Makes predictions on test data
    - Calculates accuracy
    - Stores sample images along with predictions for visualization