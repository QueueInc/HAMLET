#!/bin/bash
# IMPROVED
python automl/post_processor/etl.py --input-folder improved_1000 --output-folder improved_1000 --budget 1000
python automl/post_processor/etl.py --input-folder improved_1000 --output-folder improved_500 --budget 500
python automl/post_processor/etl.py --input-folder new_improved_500 --output-folder new_improved_500 --budget 500
python automl/post_processor/etl.py --input-folder new_improved_seed_1000 --output-folder new_improved_seed_1000 --budget 1000
python automl/post_processor/etl.py --input-folder new_improved_seed_1000 --output-folder new_improved_seed_500 --budget 500
python automl/post_processor/etl.py --input-folder new_improved_svc_1000 --output-folder new_improved_svc_1000 --budget 1000
python automl/post_processor/etl.py --input-folder new_improved_svc_1000 --output-folder new_improved_svc_500 --budget 500
python automl/post_processor/etl.py --input-folder new_improved_svc_seed_500 --output-folder new_improved_svc_seed_500 --budget 500
python automl/post_processor/etl.py --input-folder new_improved_svc_it_1000 --output-folder new_improved_svc_it_1000 --budget 1000
python automl/post_processor/etl.py --input-folder new_improved_svc_data_500 --output-folder new_improved_svc_data_500 --budget 500
python automl/post_processor/etl.py --input-folder new_improved_svc_data_1000 --output-folder new_improved_svc_data_1000 --budget 1000

# NOLIMIT
python automl/post_processor/etl.py --input-folder nolimit_1000 --output-folder nolimit_1000 --budget 1000
python automl/post_processor/etl.py --input-folder nolimit_1000 --output-folder nolimit_500 --budget 500
python automl/post_processor/etl.py --input-folder new_nolimit_1000 --output-folder new_nolimit_1000 --budget 1000
python automl/post_processor/etl.py --input-folder new_nolimit_1000 --output-folder new_nolimit_500 --budget 500
python automl/post_processor/etl.py --input-folder new_nolimit_svc_1000 --output-folder new_nolimit_svc_1000 --budget 1000
python automl/post_processor/etl.py --input-folder new_nolimit_svc_1000 --output-folder new_nolimit_svc_500 --budget 500
python automl/post_processor/etl.py --input-folder new_nolimit_seed_500 --output-folder new_nolimit_seed_500 --budget 500
python automl/post_processor/etl.py --input-folder new_nolimit_svc_seed_500 --output-folder new_nolimit_svc_seed_500 --budget 500

