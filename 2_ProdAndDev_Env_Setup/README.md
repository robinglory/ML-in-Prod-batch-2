### Anaconda
#### Anaconda Commands
```bash'
conda -h
conda env list
conda activat ths
conda create -n test python=3.10
conda remove -n test --all
```


Check libs and create sample requirements.txt
```bash
pip list
pip freeze > requirements.txt
```


### Pipenv with  BASE python version
Solving libs dependencies
```bash
pip install pipenv

pipenv install
# install in dev
pipenv install --dev seaborn

# install in prod
pipenv install tensorflow==2.18.0

pipenv graph

```




### Pipenv with Specific python version
```bash

conda activate dev_env

which python
# /opt/anaconda3/envs/dev_env/bin/python
pipenv --rm

pipenv install --python "/opt/anaconda3/envs/dev_env/bin/python"


# Locate the project:
pipenv --where


# Locate the virtualenv:
pipenv --venv

# locate the python interpreter:
pipenv --py

```



Generate pipenv to requirements.txt
```bash

pipenv run python test.py
pipenv run pip freeze > requirements.txt
```


#### Pyproject toml
```bash
pip install poetry
poetry --version
poetry init

# Add Dependencies
#To add a production dependency:
poetry add requests

#To add a development dependency:
poetry add --dev pytest

# install dependencies
poetry install

# run project
poetry run python tf_version.py


```