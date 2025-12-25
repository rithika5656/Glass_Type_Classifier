#  Glass Type Classifier

A Machine Learning project to classify different types of glass based on their chemical composition using **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** algorithms.

## PS

Different types of glass are used for various purposes such as windows, containers, tableware, and headlights. Manually identifying the type of glass using chemical composition is difficult and time-consuming. This project automates glass classification using Machine Learning.

### Chemical Features
- **RI** - Refractive Index
- **Na** - Sodium (%)
- **Mg** - Magnesium (%)
- **Al** - Aluminum (%)
- **Si** - Silicon (%)
- **K** - Potassium (%)
- **Ca** - Calcium (%)
- **Ba** - Barium (%)
- **Fe** - Iron (%)

### Glass Types
| Type | Description |
|------|-------------|
| 1 | Building Windows (Float Processed) |
| 2 | Building Windows (Non-Float) |
| 3 | Vehicle Windows (Float) |
| 5 | Containers  |
| 6 | Tableware |
| 7 | Headlamps |

> *Note: Type 4 (Vehicle Windows Non-Float) is not present in the dataset*

##  Algorithms Used

### 1. K-Nearest Neighbors (KNN)
- Classifies based on majority vote of k nearest neighbors
- Optimal k value determined through cross-validation
- Simple and interpretable

### 2. Support Vector Machine (SVM)
- Finds optimal hyperplane for class separation
- Multiple kernels tested (Linear, RBF, Polynomial)
- Effective for high-dimensional data

##  Features

- Data preprocessing and exploratory analysis
- Feature scaling using StandardScaler
- Hyperparameter tuning for optimal performance
- Model comparison and evaluation
- Comprehensive visualizations
- **Interactive Glass Classifier** with:
  - Image upload capability
  - Chemical property input sliders
  - Sample data buttons for quick testing
  - Real-time predictions from both models

## 📈 Results

| Model | Accuracy |
|-------|----------|
| KNN | ~70% |
| SVM | ~68% |

*Actual results may vary based on random state*

### Using the Interactive Classifier
1. **Upload an image** (optional) - for visual reference
2. **Enter chemical properties** using sliders OR click sample buttons
3. **Click "CLASSIFY GLASS TYPE"** to get predictions

## 🔧 Technologies Used

- **Python 3.8+**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization
- **Scikit-learn** - Machine Learning
- **ipywidgets** - Interactive UI
- **Pillow** - Image processing



## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Glass Identification Dataset
- Scikit-learn documentation and community

---

⭐ **If you found this project helpful, please give it a star!** ⭐
