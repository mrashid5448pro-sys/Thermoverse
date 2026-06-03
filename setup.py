from setuptools import setup, find_packages

setup(
    name="thermoverse",
    version="1.0.0",
    author="mrashid5448pro-sys",
    description="Materials property prediction using ML (thermodynamics + mechanical properties)",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "catboost",
    ],
)
