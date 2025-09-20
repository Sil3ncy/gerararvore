from setuptools import setup, find_packages

setup(
    name="geraravore",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "graphviz"
    ],
    description="Implementações simples de ID3, C4.5 e CART",
    author="Diego Feitosa Ferreira Dos Santos",
    url="https://github.com/Sil3ncy/gerararvore",
)
