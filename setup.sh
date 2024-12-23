#!/usr/bin/bash

python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip

pip3 install tqdm
pip3 install numpy
pip3 install causaldag
pip3 install seaborn
pip3 install optimaladj