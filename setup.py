import setuptools


if __name__ == "__main__":
    setuptools.setup(
        name="jabas",
        version="1.0.0",
        author="The JABAS Authors at UNIST",
        author_email="rugyoon@gmail.com",
        description="JABAS: Joint Adaptive Batching and Automatic Scaling for DNN Training on Heterogeneous GPUs",
        url="https://github.com/unist-ssl/JABAS",
        packages=setuptools.find_packages(include=["jabas", "jabas.*"]),
        python_requires='>=3.6',
        install_requires=[]
    )
