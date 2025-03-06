import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as open_file:
    install_requires = open_file.read()


setuptools.setup(
    name='lscaleuq',
    version='1.0.0',
    url='https://github.com/ykrmm/large_scale_uq',
    packages=setuptools.find_packages(),
    author='Yannis Karmim & Marc Lafon',
    author_email='lafon.ma.ml@gmail.com',
    description='Large Scale Uncertainty Pretraining for Vision Language Models',
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