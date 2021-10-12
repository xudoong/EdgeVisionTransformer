#!/bin/bash

pip install -r requirements.txt
cd vendor/huggingface_transformers
python setup.py install --user
cd ../..