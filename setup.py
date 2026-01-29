from setuptools import setup, find_packages

setup(
    name="glass_classifier",
    version="0.1.0",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'joblib',
        'streamlit'
    ],
    author="Antigravity",
    description="A package for classifying glass types.",
)
