from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="Epistemic_CP",
    version="1.0.0",
    description="Epistemic Conformal Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=".",
    author="Anonymous",
    author_email=".",
    packages=["Epistemic_CP"],
    license="MIT",
    keywords=[
        "prediction regions",
        "conformal prediction",
        "conditional coverage",
        "epistemic uncertainty",
    ],
    install_requires=[
        "numpy>=1.26.4",
        "scikit-learn==1.5.1",
        "scipy==1.14.1",
        "matplotlib==3.9.2",
        "tqdm==4.66.5",
        "torch>=2.3.1",
        "gpytorch==1.13",
        "pymc",
    ],
    zip_safe=False,
)
