# 🔬 Glass Type Classifier

A Machine Learning project and Web Application to classify different types of glass based on their chemical composition using **K-Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)** algorithms.

## 🚀 Enhancements & Features

This project has been significantly refactored from a simple notebook into a production-ready Python package with a web interface.

### Key Improvements (40+ Changes)
1. **Modular Architecture**: Codebase refactored into `src/glass_classifier` package.
2. **Separation of Concerns**: Dedicated modules for `data_loader`, `preprocessing`, `models`, `train`, `evaluate`, `predict`, `viz`.
3. **Configuration Management**: Centralized `config.py` for easy tuning.
4. **Web Application**: Interactive Streamlit app (`src/web_app/app.py`) for real-time predictions.
5. **Interactive Analysis**: Data exploration and visualization directly in the web app.
6. **Robust Data Loading**: Smarter CSV handling (header detection, path resolution).
7. **Pipeline Design**: Training pipeline separated from script logic.
8. **Model Persistence**: Automatic saving/loading of models using `joblib`.
9. **Scalability**: `StandardScaler` integration for proper feature scaling.
10. **Type Hinting & Docstrings**: Added documentation to all functions.
11. **Testing Suite**: Added `tests/` directory with pytest support.
12. **Dependency Management**: Added `requirements.txt`.
13. **Package Setup**: Added `setup.py` for installation.
14. **Error Handling**: Improved error messages for missing files/models.
15. **Visualization Library**: Custom plotting functions in `visualization.py`.
16. **Dynamic Predictions**: Web app accepts user input for all 9 chemical features.
17. **Model Comparison**: Side-by-side display of KNN and SVM simple predictions.
18. **CI/CD Ready**: Structure supports automation.
19. **Code Quality**: Adherence to PEP 8 standards (variable naming, imports).
20. **Extensibility**: Easy to add new models.
21. **Automated Data Download**: Automatically downloads UCI dataset if missing.
22. **Robust Preprocessing**: Added outlier handling and RobustScaler support.
23. **More Models**: Added Random Forest, Gradient Boosting, and Logistic Regression.
24. **Feature Importance**: Visualized feature importance for tree-based models.
25. **Evaluation Metrics**: Saves detailed metrics to JSON files.
26. **Cross Validation**: Added cross-validation support.
27. **Logging**: Integrated python logging for better debugging.
28. **Reproducibility**: Global random seed setting.
29. **System Info**: Utility to log system environment.
30. **Unit Tests**: comprehensive unit tests for models and data loader.


## 📂 Project Structure

```
GLASS IDENTIFICATION/
├── src/
│   ├── glass_classifier/   # Core Package
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── models.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── utils.py
│   │   └── visualization.py
│   └── web_app/            # Streamlit App
│       └── app.py
├── tests/                  # Unit Tests
├── glass.csv               # Dataset
├── models/                 # Saved Models (generated)
├── requirements.txt
├── setup.py
└── README.md
```

## 🛠️ Installation & Usage

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Web App**
   ```bash
   streamlit run src/web_app/app.py
   ```
   Open your browser to the URL shown (usually http://localhost:8501).

3. **Train Models**
   - Go to the **Model Training** page in the web app.
   - Click **Train Models**.
   - Models will be saved to the `models/` directory.

4. **Run Tests**
   ```bash
   pytest
   ```

## 📊 Dataset Info

- **Source**: UCI Machine Learning Repository
- **Instances**: 214
- **Features**: 9 chemical attributes (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe)
- **Target**: Glass Type (1-7)

## 🤝 Contributing

Feel free to fork and submit pull requests!
