import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as open_file:
    install_requires = open_file.read()


setuptools.setup(
    name='vilu',
    version='1.0.0',
    url='Anonymized',
    packages=setuptools.find_packages(),
    author='Anonymized',
    author_email='Anonymized',
    description='ViLU: Learning Vision-Language Uncertainties for Failure Prediction',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=install_requires,
)