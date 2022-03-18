from setuptools import setup

setup(
    name='cvdea',
    version='0.1',
    description='Cross-validated DEA',
    url='',
    author='Miriam Esteve and Juan Aparicio',
    author_email='miriam.estevec@umh.es',
    packages=['cvdea'],
    install_requires=['numpy', 'pandas', 'graphviz', 'docplex', "matplotlib"],
    license='AFL-3.0',
    zip_safe=False
)