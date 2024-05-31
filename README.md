# HAMLET: Human-centered AutoMl via Logic and argumEnTation

This work is the implementation of the framework proposed in: [HAMLET: a framework for Human-centered AutoML via Structured Argumentation](https://authors.elsevier.com/sd/article/S0167-739X(22)00439-3).

If you are interested in reproducing the experiments performed in it, please refer to the dedicated [GitHub repository](https://github.com/QueueInc/HAMLET-FGCS2022).

## Philosophy

In the last decades, we have witnessed an exponential growth in the field of AutoML.
Unfortunately, this lead AutoML to be just another black-box that can be barely open.
We claim that the user / data scientist has the right and the duty to revise and supervise the Machine Learning (ML) / Data Mining (DM) process.

HAMLET is a Human-centered AutoML framework that leverages Logic and Argumentation to:
- inject knowledge and contraints through a intuitive logical language into a Logical Knowledge Base (LogicalKB);
- represent (user) inputs and (AutoML) outputs in a uniform medium that is both human- and machine-readable (Problem Graph);
- discover insights through a recommendation mechanism atop the AutoML outputs;
- deal with possible arising inconsistencies in the knowledge.

HAMLET is inspired to the well-known and standard process model CRISP-DM (CRoss-Industry Standard Process for Data Mining). 
Iteration after iteration, the knowledge is augmented and data scientists and domain experts can work in a close cooperation towards the final solution.

![dymmymodel](https://user-images.githubusercontent.com/41596745/209567103-f06febea-0bbb-4c1c-96fb-cfff2ec9d53c.png)

# HAMLET in action!

HAMLET leverages:
- [Arg-tuProlog](https://ceur-ws.org/Vol-2710/paper4.pdf), a Kotlin implementation of the Argumentation framework;
- [Microsoft FLAML](https://microsoft.github.io/FLAML/), a python implementation of the state-of-the-art Blend Search (a Bayesian Optmization variant that takes into consideration the cost of possible solution);
- [Scikit-learn](https://scikit-learn.org/stable/), the well-known python framework that offers plenty of ML algorithm implementations.

## REQUIREMENTS

- Docker
- Java >= 11.0

## EXECUTION


```
java -jar hamlet-1.0.0-all.jar [workspace_path] [dataset_id] [optimization_metric] [optimization_mode] [n_configurations] [time_budget] [optimization_seed] [debug_mode] [knowledge_base_path]
```

- **[workspace_path]**: absolute path to the file system folder containing the workspace (i.e., where to save the results); if it does not exist, a new workspace is created, otherwise, the previous run is resumed.
- **[dataset_id]**: OpenML id of the dataset to analyze (due to some OpenML API disservices, we pre-downloaded the datasets, thus we support only a specific suite: the [OpenML CC-18 suite](https://www.openml.org/search?type=study&study_type=run&id=99)).
- **[optimization_metric]**: a string of the metric name to optimize (choose among the [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter), e.g, ```balanced-accuracy```).
- **[optimization_mode]**: a string in ['min', 'max'] to specify the objective as minimization or maximization.
- **[n_configurations]**: an integer of the number of configurations to try in the optimization of each iteration.
- **[time_budget]**: the time budget in seconds given to the optimization of each iteration.
- **[optimization_seed]**: seed for reproducibility.
- **[debug_mode]**: a string in ['true', 'false'] to specify HAMLET execution in debug or release mode. In debug mode, the Docker container is built from the local sources; otherwise the released Docker image is downloaded.
- **[knowledge_base_path]** (OPTIONAL): file system path to an HAMLET knowledge base. If provided, HAMLET is run in console (with no GUI) mode and the theory is leveraged; otherwise HAMLET GUI is launched.

## Functioning

Once you run HAMLET with a specific dataset and a specific metric to optimize, at the top of the window, HAMLET allows to encode both the AutoML search space and the user-defined constraints into the LogicalKB (see the next section for the accepted syntax).

Two ready-to-use LogicalKB can be found in the resources folder of this repository:
- ```kb.txt``` is a knowledge base containing the search space leveraged in our experiments;
- ```pkb.txt``` (PreliminaryKB) is a knowledge base containing the search space along with some suggested constraints (discovered in the paper [Data pre-processing pipeline generation for AutoETL](https://www.sciencedirect.com/science/article/abs/pii/S0306437921001514)).

For the sake of brevity, follows an example with a simplier LogicalKB:

<img width="960" alt="hamlet_gui" src="https://user-images.githubusercontent.com/41596745/209572069-4e63d9b5-a88d-405b-bf01-cc1a02cd1812.png">

Intuitively, we specify the scheme of the ML pipeline that we want to build, step by step.
In the example at hand, we have:
- a Data Pre-processing step for Discretization;
- a Data Pre-processing step for Normalization;
- a Modeling step for Classification (the task we want to address).

Then, we have the implementations and the hyper-parameter domains of each step:
- KBins for Discretization, with a integer parameter k_bins that ranges from 3 to 8;
- StandardScaler for Normalization, with no parameter;
- Decision Tree and K-Nearest Neighbors for Classification, with -- respectively -- an integer parameter max_depth that ranges from 1 to 5 and an integer parameter n_neighbors that ranges from 3 to 20.

Finally, we have a user-defined constraints (```c1```): forbid Normalization for Decision Tree.

By hitting the ```Compute Graph``` button, the Argumentation framework is called to process the encoded LogicalKB.
The Problem Graph is visualized at the bottom-right corner.
Each node of this Argumentation graph (called arguments) represent a specific portion of search sub-space, the legend is visualized at the bottom-left corner.
For instance:
- A1, A3, A5, A7, and A9 represent all the possible pipelines for the Decision Tree algorithm;
- A2, A4, A6, A8, and A10 represent all the possible pipelines for the K-Nearest Neighbor algorithm.

Besides, each constraint is represented as an argument as well.
Indeed, the node (argument) A0 represent the user-defined constraint ```c1```.

Edges are attacks from an argument to another (```c1``` attacks exactly the pipelines in which we have Normalization along with the Decision Tree).

By hitting the ```Run AutoML``` button, HAMLET triggers FLAML to explore the encoded search space, taking also in consideration the specified constraints (discouraging the exploration in those particular sub-spaces).

At the end of the optimization, the user can switch to the ```Data``` tab to go through all the explored configurations:

<img width="956" alt="hamlet-gui-data2" src="https://user-images.githubusercontent.com/41596745/209576316-92bb528a-b180-4b61-83fd-621a3f8e3589.png">

As to the last tab ```AutoML arguments```, we can see reccomendations of constraints, mined from the AutoML output:

<img width="959" alt="hamlet-gui-rules2" src="https://user-images.githubusercontent.com/41596745/210392351-13491f27-e07f-4e3e-a012-4f2e3692bc52.png">

We think at this process as an *argument* between the data scientist and the AutoML tool.
The data scientist can consider the arguments at hand, and encode them into the LogicalKB.


At this point, the next iteration can be performed.

## LogicalKB syntax

We committed in developing a logical language as intuitive as possible:
- ```step(S).``` specifies a step ```S``` of the pipeline, with ```S``` in [```discretization```, ```normalization```, ```rebalancing```, ```imputation```, ```features```, ```classification```]
- ```operator(S, O).``` specifies an operator ```O``` for the step ```S```, with ```O``` in [```kbins```, ```binarizer```, ```power_transformer```, ```robust_scaler```, ```standard```,  ```minmax```, ```select_k_best```, ```pca```, ```simple_imputer```, ```iterative_imputer```, ```near_miss```, ```smote```]
- ```hyperparameter(O, H, T).``` specifies an hyper-parameter ```H``` for the operator ```O``` with type ```T```, ```H``` can be every hyper-parameter name of the chosen Scikit-learn operator ```O```, ```T``` is chosen accordingly and has to be in [```randint```, ```choice```, ```uniform```]
- ```domain(O, H, D).``` specifies the domain ```D``` of the hyper-parameter ```H``` of the operatore ```O```, ```D``` is an array in ```[ ... ]``` brackets containing the values that the hyper-parameter ```H``` can assume (in case of ```randint``` and ```uniform```, the array has to contain just two elements: the boundary of the range)
- ```id :=> mandatory_order([S1, S2], O1).``` specifies a ```mandatory_order``` constraint: the step ```S1``` has to appear before the step ```S2``` when occurring the operator ```O1``` of the task step (in this implementation we support only ```classification``` task); it is possible to put ```classification``` instead of ```O1```, this will apply the constraint for each ```classification``` operators
- ```id :=> mandatory([S1, S2, ...], O1).``` specifies a ```mandatory``` constraint: the steps ```[S1, S2, ...]``` are mandatory when occurring the operator ```O1``` of the task step (in this implementation we support only ```classification``` task); if the array of the steps is empty, the constraint specifies only that O1 is mandatory (with or withour Data Pre-processing steps)
- ```id :=> forbidden([S1, S2, ...], O1).``` specifies a ```forbidden``` constraint: the steps ```[S1, S2, ...]``` are forbidden when occurring the operator ```O1``` of the task step (in this implementation we support only ```classification``` task); if the array of the steps is empty, the constraint specifies only that O1 is forbidden (with or withour Data Pre-processing steps)


# Debug

/usr/local/bin/python /home/automl/main.py -dataset 31 -metric balanced_accuracy -fair_metric demographic_parity -mode max -batch_size 125 -time_budget 900 -input_path /home/results_fairness/pkb_cluster/automl/input/automl_input_1.json -output_path /home/results_fairness/pkb_cluster/automl/output/automl_output_1.json -seed 42

Notes
- fair_metric is new
- 31 is the german-credit dataset