#!/bin/bash
sudo ./scripts/run_hamlet.sh results/baseline_1000_ext balanced_accuracy max 500 3600 0.2.17 1 $(pwd)/resources/kb_extended.txt
sudo ./scripts/run_hamlet.sh results/pkb_1000_ext balanced_accuracy max 500 3600 0.2.17 1 $(pwd)/resources/pkb_extended.txt
sudo ./scripts/run_hamlet.sh results/ika_1000_ext balanced_accuracy max 125 900 0.2.17 4 $(pwd)/resources/kb_extended.txt
sudo ./scripts/run_hamlet.sh results/pkb_ika_1000_ext balanced_accuracy max 125 900 0.2.17 4 $(pwd)/resources/pkb_extended.txt