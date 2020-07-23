from setuptools import setup

setup(
    name='song-cutter',
    version='0.1.0',
    packages=['Pipeline', 'Pipeline/DataGenerator', 'Pipeline/FeatureExtractor', 'Pipeline/Preprocessing', 'Pipeline/Segmentation'],
    url='https://github.com/korotetskiiba/song-cutter',
    license='Apache License 2.0',
    author='Team audio segmentation',
    author_email='',
    description='Audio segmentation'
)
