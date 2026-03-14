#!/bin/bash
apt-get update && apt-get install -y libomp-dev gcc g++
python -m pip install --upgrade pip wheel setuptools
pip install --only-binary=all catboost==1.5.0
pip install --only-binary=all -r requirements.txt
