#!/bin/bash
wget https://github.com/QueueInc/HAMLET/releases/download/$5/hamlet-$5-all.jar
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openml
python -m pip install tqdm
python -m pip install pandas
python automl/run_baseline.py --workspace $1 --metric $2 --mode $3 --batch_size $4 --time_budget $5 --version $6 --range $7 --num_tasks $8 --iterations $9
deactivate