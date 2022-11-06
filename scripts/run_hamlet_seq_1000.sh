#!/bin/bash
sudo ./scripts/run_hamlet.sh results/baseline_500_ext balanced_accuracy max 1000 7200 0.2.17 1 $(pwd)/resources/kb_extended.txt
sudo ./scripts/run_hamlet.sh results/pkb_500_ext balanced_accuracy max 1000 7200 0.2.17 1 $(pwd)/resources/pkb_extended.txt
sudo ./scripts/run_hamlet.sh results/ika_500_ext balanced_accuracy max 250 1800 0.2.17 4 $(pwd)/resources/kb_extended.txt
sudo ./scripts/run_hamlet.sh results/pkb_ika_500_ext balanced_accuracy max 250 1800 0.2.17 4 $(pwd)/resources/pkb_extended.txt