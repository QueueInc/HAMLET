# REQUIREMENTS
- Docker
- Python >=3.7
- Java >=11.0

# RUN HAMLET

    java -jar hamlet-0.2.10-all.jar [workspace_path] [dataset_id] [optimization_metric] [optimization_mode] [n_configurations] [time_budget] [optimization_seed] [debug_mode] [knowledge_base_path]

- **[workspace_path]**: file system folder cotaining the workspace (if it does not exist, a new workspace is created; otherwise, the previous run is resumed).
- **[dataset_id]**: OpenML id of the dataset to analyze.
- **[optimization_metric]**: a string of the metric name to optimize (choose among the [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)).
- **[optimization_mode]**: a string in ['min', 'max'] to specify the objective as minimization or maximization.
- **[n_configurations]**: an integer of the number of configurations to try in the optimization.
- **[time_budget]**: the time budget in seconds given to the optimization.
- **[optimization_seed]**: seed for reproducibility.
- **[debug_mode]**: a string in ['true', 'false'] to specify HAMLET execution in debug or release mode. In debug mode, the Docker container is built from the local sources; otherwise the released Docker image is downloaded.
- **[knowledge_base_path]** (OPTIONAL): file system path to an HAMLET knowledge base. If provided, HAMLET is run in console (with no GUI) mode and the theory is leveraged; otherwise HAMLET GUI is launched.


# REPRODUCING PAPER EXPERIMENTS

- Clone the repository and give permission

      sudo git clone https://github.com/QueueInc/HAMLET.git
      cd HAMLET
      sudo chmod 777 scripts/run_hamlet.sh

- Run the baseline (AutoML explores 1000 configurations in a single run with a time budget of 7200 seconds).

      sudo ./scripts/run_hamlet.sh results/baseline balanced_accuracy max 1000 7200 0.2.10 1 $(pwd)/resources/kb.txt

- Run HAMLET with PKB (Preliminary Knowledge Base) settings, HAMLET starts with a preliminary LogicalKB constraining the search space from the first iteration, and no rule mining is applied.

      sudo ./scripts/run_hamlet.sh results/baseline balanced_accuracy max 1000 7200 0.2.10 1 $(pwd)/resources/pkb.txt

- Run HAMLET with IKA (Iterative Knowledge Augmentation) settings, HAMLET starts with an empty LogicalKB, and the rules recommended after each run are applied to extend the LogicalKB.

      sudo ./scripts/run_hamlet.sh results/baseline balanced_accuracy max 250 1800 0.2.10 4 $(pwd)/resources/kb.txt

- Run HAMLET with PKB + IKA settings, HAMLET starts with a preliminary LogicalKB, and the rules recommended after each run are applied to extend the LogicalKB.

      sudo ./scripts/run_hamlet.sh results/baseline balanced_accuracy max 250 1800 0.2.10 4 $(pwd)/resources/pkb.txt