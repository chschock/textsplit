from setuptools import setup, find_packages

setup(
    name='textsplit',
    version=0.9,
    description='Segment documents into coherent parts using wordembeddings.',
    url='https://github.com/chschock/textsplit',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='Christoph Schock',
    author_email='chschock@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'nose>=1.3.7',
        'numpy>=1.13.1',
        'nose>=1.3.7',
    ],
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'License :: MIT License',
        'Operating System :: Linux',
        'Topic :: NLP',
    ),
    keywords='nlp text segmentation paragraph embeddings',
)
