from setuptools import setup, find_packages

setup(
    name='sharelock',
    version='1.0',
    author='Jona Ruthardt',
    author_email='jona@ruthardt.de',
    description='ShareLock - an ultra-lightweight CLIP-like vision-language model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/JonaRuthardt/ShareLock',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "pytorch-lightning",
        "transformers",
        "omegaconf",
        "tqdm",
        "tensorboard",
        "datasets",
        "featureutils @ git+https://github.com/JonaRuthardt/featureutils.git",
    ],
    include_package_data=True,
    zip_safe=False,
)