# REQUIREMENTS
- Docker
- Java >=11.0

# RUN HAMLET

    java -jar hamlet-1.0.0-all.jar [workspace_path] [dataset_id] [optimization_metric] [optimization_mode] [n_configurations] [time_budget] [optimization_seed] [debug_mode] [knowledge_base_path]

- **[workspace_path]**: file system folder cotaining the workspace (if it does not exist, a new workspace is created; otherwise, the previous run is resumed).
- **[dataset_id]**: OpenML id of the dataset to analyze.
- **[optimization_metric]**: a string of the metric name to optimize (choose among the [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)).
- **[optimization_mode]**: a string in ['min', 'max'] to specify the objective as minimization or maximization.
- **[n_configurations]**: an integer of the number of configurations to try in the optimization.
- **[time_budget]**: the time budget in seconds given to the optimization.
- **[optimization_seed]**: seed for reproducibility.
- **[debug_mode]**: a string in ['true', 'false'] to specify HAMLET execution in debug or release mode. In debug mode, the Docker container is built from the local sources; otherwise the released Docker image is downloaded.
- **[knowledge_base_path]** (OPTIONAL): file system path to an HAMLET knowledge base. If provided, HAMLET is run in console (with no GUI) mode and the theory is leveraged; otherwise HAMLET GUI is launched.