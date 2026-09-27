from setuptools import setup, find_packages

setup(
    name="delivery_time_prediction",
    version="1.0.0",
    description="E-Commerce Delivery Time Prediction System using Brazilian Olist Dataset",
    author="Lavindi Tharunya",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "joblib>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "pydantic>=2.0.0",
        "gradio>=4.0.0",
    ]
)
