from setuptools import setup, find_packages

setup(
    name='ML_evaluator',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    entry_points={
        'console_scripts': [
            'ML_evaluator=ML_evaluator:main',
        ],
    },
    author='Your Name',
    author_email='racerpoet1@gmail.com',
    description='A library to evaluate multiple machine learning models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KPMOKGOPA/ML_evaluator',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
