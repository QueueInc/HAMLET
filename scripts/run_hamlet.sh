#!/bin/bash
wget https://github.com/QueueInc/HAMLET/releases/download/$5/hamlet-$5-all.jar
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openml
python -m pip install tqdm
python -m pip install pandas
python automl/run_hamlet.py --workspace $1 --metric $2 --mode $3 --batch_size $4 --version $5 --range $6 --num_tasks $7 --iterations $8
deactivate