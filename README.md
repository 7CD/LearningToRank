# LearningToRank

Participation in kaggle [competition](https://www.kaggle.com/c/learning-to-rank-made-fall-2019).

[https://github.com/7CD/LearningToRank/blob/dev/src/ranking/model.py](Implementatoin of LambdaMART) ranking algorithm following 
[*J.C. Burges. From RankNet to LambdaRank to LambdaMART: An overview. Technical Report MSR-TR-2010-82, Microsoft Research, 2010.*](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf)


# Project Structure
------------------------
```
    .
    ├── data
    │   ├── processed               <- processed data
    │   └── raw                     <- original unmodified/raw data
    ├── models                      <- folder for ML models
    ├── notebooks                   <- Jupyter Notebokos (ingored by Git)
    ├── reports                     <- folder for experiment reports
    ├── src                         <- source code for modules & pipelines
    └── README.md
```

## Preparation

### 1. Fork / Clone this repository

```bash
git clone https://github.com/7CD/LearningToRank.git
cd LearningToRank
```

### 2. Create a `dev` branch and make it a default branch 
```bash
git checkout dev
```
 
### 3. Create and activate virtual environment

Create virtual environment named `myvenv` (you may use other name)
```bash
python3 -m venv myvenv
echo "export PYTHONPATH=$PWD" >> myvenv/bin/activate
source myvenv/bin/activate
```
Install python libraries

```bash
pip install -r requirements.txt
```
Add Virtual Environment to Jupyter Notebook

```bash
python -m ipykernel install --user --name=myvenv
``` 

And install your project in editable mode:

```bash
python -m pip install -e .
``` 

### 4. Create kaggle API token.

In order to use the Kaggle’s public API, you must first authenticate using an API token. From the site header, click on your user profile picture, then on “My Account” from the dropdown menu. This will take you to your account settings at https://www.kaggle.com/account. Scroll down to the section of the page labelled API:

To create a new token, click on the “Create New API Token” button. This will download a fresh authentication token onto your machine.

If you are using the Kaggle CLI tool, the tool will look for this token at ~/.kaggle/kaggle.json on Linux, OSX, and other UNIX-based operating systems, and at C:\Users<Windows-username>.kaggle\kaggle.json on Windows. If the token is not there, an error will be raised. Hence, once you’ve downloaded the token, you should move it from your Downloads folder to this folder.
[(https://www.kaggle.com/docs/api)](https://www.kaggle.com/docs/api)

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
``` 

## 5. Run Jupyter Notebook

Jupyter Notebooks in `notebooks/` directory are for example only. 
To remove them (recommended) from `git` version control run: 

1 - Add the following string to `.gitignore`
```.gitignore
notebook/*
git add .gitignore
git commit -m "Update .gitignore: add notebooks/* " 
```
2 - Remove notebooks from the Git index and commit changes
```bash
git rm --cached notebooks/*
git commit -m "Unstage notebooks" 
```
Note: this will remove files from the Git index only! Files won’t be deleted from the disk

```bash
jupyter notebook
```

## 6. Pipline

Run parts of pipeline from console, e.g.:

```bash
python src/pipelines/data_load.py --config=params.yaml
```
