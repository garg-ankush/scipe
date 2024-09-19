from setuptools import setup, find_packages

setup(
    name="scipe",
    version="2.0.0",
    packages=find_packages(exclude=("build", "example-data", "scipe.egg-info")),
    author="",
    author_email="",
    description="Systematic Chain Improvement and Problem Evaluation",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    python_requires=">=3.9",
    install_requires=[
        "langchain-anthropic==0.1.16",
        "langchain-core==0.2.28",
        "langgraph==0.1.19",
        "ipykernel==6.29.5",
        "pandas==2.2.2",
        "python-dotenv==1.0.1",
        "openpyxl==3.1.5",
        "litellm==1.46.5"
    ],
)
