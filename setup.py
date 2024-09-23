from setuptools import setup

setup(
    name="scipe",
    version="1.0.0",
    package_dir={"scipe": "src"},
    packages=["scipe"],
    author="Ankush Garg",
    author_email="ankush-garg@berkeley.edu",
    description="Systematic Chain Improvement and Problem Evaluation",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    python_requires=">=3.9",
    install_requires=[
        "langchain-anthropic==0.1.16",
        "langchain-core==0.2.28",
        "langgraph==0.1.19",
        "ipykernel==5.5.6",
        "pandas==2.1.4",
        "python-dotenv==1.0.1",
        "openpyxl==3.1.5",
        "litellm==1.46.5"
    ],
    include_package_data=True
)
