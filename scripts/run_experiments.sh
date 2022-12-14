sudo ./scripts/run_hamlet.sh results/hamlet balanced_accuracy max 500 3600 1.0.0 1 $(pwd)/resources/kb.txt
sudo ./scripts/run_hamlet.sh results/hamlet balanced_accuracy max 500 3600 1.0.0 1 $(pwd)/resources/pkb.txt
sudo ./scripts/run_hamlet.sh results/hamlet balanced_accuracy max 125 900 1.0.0 4 $(pwd)/resources/kb.txt
sudo ./scripts/run_hamlet.sh results/hamlet balanced_accuracy max 125 900 1.0.0 4 $(pwd)/resources/pkb.txt
sudo ./scripts/run_comparison.sh h2o_500 3600 results/h2o
sudo ./scripts/run_comparison.sh auto_sklearn_500 3600 results/auto_sklearn
python automl/post_processor/etl.py --input-folder hamlet --output-folder hamlet --budget 500