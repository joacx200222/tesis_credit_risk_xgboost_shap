from setuptools import setup, find_packages

setup(
    name="tesis_credit_risk_xgboost_shap",
    version="1.0.0",
    description="Modelo de evaluación de riesgo crediticio para microfinanzas mediante XGBoost e IA Explicable (SHAP)",
    author="Joaquín Ignacio Bernales Demichelli",
    author_email="20190255@aloe.ulima.edu.pe",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "shap>=0.44.0",
        "optuna>=3.5.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.50.0",
    ],
)
