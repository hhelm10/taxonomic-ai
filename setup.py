from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    readme = f.read()
VERSION = '0.0.1'
NAME = 'taxi'
setup(
    name=NAME,
    packages=find_packages(exclude=['tests', 'misc', 'asset']),
    version=VERSION,
    description='Classifying and cateogorizing collections of generative moels.',
    url='https://github.com/hhelm10/taxonomi-ai',
    keywords=['generative models', 'embeddings', 'populations of models'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Hayden Helm',
    author_email='hayden@helivan.io',
    install_requires=[
        "graspologic",
        "torch",
        "transformers"
    ],
)
