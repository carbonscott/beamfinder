import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="behenate_net",
    version="1.0.0",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="...",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/beamfinder.git",
    keywords = [],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
