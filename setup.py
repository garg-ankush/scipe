from setuptools import setup # type: ignore

with open("README.md", "r") as f:
    description = f.read()

setup(
    name="scipe",
    version="1.0.1",
    package_dir={"scipe": "src"},
    packages=["scipe"],
    author="Ankush Garg",
    author_email="ankush-garg@berkeley.edu",
    description="Systematic Chain Improvement and Problem Evaluation",
    long_description=description,
    long_description_content_type="text/markdown",
    url="",
    python_requires=">=3.9",
    install_requires=[
        "langchain-anthropic==0.1.16",
        "langchain-core==0.2.28",
        "langgraph==0.1.19",
        "ipykernel==5.5.6",
        "pandas==2.2.2",
        "python-dotenv==1.0.1",
        "openpyxl==3.1.5",
        "litellm==1.46.5"
    ],
    extras_requires={
        "dev": ["twine==5.1.1"]
    },
    include_package_data=True
)
