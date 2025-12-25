# Glass Type Classifier 🔬

A machine learning project for glass type classification using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) algorithms. This project includes comprehensive data preprocessing, model training, evaluation, and an interactive classifier with chemical property sliders.

## Features

- 📊 **Data Preprocessing**: Load and clean glass composition data
- 🤖 **KNN Model**: K-Nearest Neighbors classifier with hyperparameter tuning
- 🎯 **SVM Model**: Support Vector Machine with multiple kernel options
- 📈 **Model Evaluation**: Detailed performance metrics and comparison
- 🎨 **Visualizations**: Confusion matrices, correlation heatmaps, and performance charts
- 🎛️ **Interactive Classifier**: Real-time predictions using ipywidgets sliders for chemical properties

## Dataset

The project uses glass identification data based on chemical composition:

**Features (Chemical Properties):**
- RI: Refractive Index
- Na: Sodium
- Mg: Magnesium
- Al: Aluminum
- Si: Silicon
- K: Potassium
- Ca: Calcium
- Ba: Barium
- Fe: Iron

**Target Variable:**
- Type: Glass type (1-7)
  - Type 1: Building Windows (Float Processed)
  - Type 2: Building Windows (Non-Float Processed)
  - Type 3: Vehicle Windows (Float Processed)
  - Type 5: Containers
  - Type 6: Tableware
  - Type 7: Headlamps

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rithika5656/Glass_Type_Classifier.git
cd Glass_Type_Classifier
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Glass_Type_Classification.ipynb`

3. Run all cells to:
   - Load and explore the dataset
   - Preprocess the data
   - Train KNN and SVM models
   - Evaluate model performance
   - Use the interactive classifier

## Interactive Classifier

The notebook includes an interactive widget that allows you to:
- Adjust sliders for each chemical property
- Get real-time predictions from both KNN and SVM models
- See if both models agree on the classification

## Model Performance

Both models are trained and compared:
- **KNN**: Optimizes the number of neighbors using cross-validation
- **SVM**: Tests different kernels (linear, RBF, polynomial) with hyperparameter tuning

The notebook provides detailed performance metrics including:
- Accuracy scores
- Confusion matrices
- Classification reports
- Model comparison visualizations

## Technologies Used

- **Python**: Core programming language
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning models and utilities
- **matplotlib & seaborn**: Data visualization
- **ipywidgets**: Interactive widgets for Jupyter notebooks
- **Jupyter Notebook**: Interactive development environment

## Project Structure

```
Glass_Type_Classifier/
├── Glass_Type_Classification.ipynb  # Main Jupyter notebook
├── glass.csv                        # Dataset
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── LICENSE                          # License file
```

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## License

This project is licensed under the terms specified in the LICENSE file.

## Author

Created by [rithika5656](https://github.com/rithika5656)
