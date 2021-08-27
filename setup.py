from os import path
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    name='embsamp',
    description="package for up/down sampling from an embedded space",
    long_description=readme,
    author="Anthony DeGennaro",
    author_email='adegennaro@bnl.gov',
    url='https://github.com/adegenna/embeddingSampler',
    python_requires='>=3.7',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    install_requires=requirements,
    license="ISC",
    classifiers=[
        'Topic :: Bayesian Optimization :: Linear Embeddings',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: ISC License',
        'Programming Language :: Python :: 3.7'
    ],
    keywords = 'bayesian optimization uncertainty quantification linear embeddings alebo',
    entry_points = { 'console_scripts': ['Package = embsamp.scripts.example_2to3d:main' ] },
)
