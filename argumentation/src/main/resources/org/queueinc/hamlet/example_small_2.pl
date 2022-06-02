s1 :=> step(discretization).
s1 :=> step(normalization).
sc1 :=> step(classification).


o1 : step(discretization) => operator(discretization, kbins).
o2 : step(discretization) => operator(discretization, binarizer).
o3 : step(normalization) => operator(normalization, power_transformer).
o4 : step(normalization) => operator(normalization, robust_scaler).
o5 : step(normalization) => operator(normalization, standard).
o6 : step(normalization) => operator(normalization, minmax).
oc1 : step(classification) => operator(classification, dt).
oc2 : step(classification) => operator(classification, knn).

h1 : operator(discretization, kbins) => hyperparameter(kbins, n_bins, randint).
h2 : operator(discretization, kbins) => hyperparameter(kbins, encode, choice).
h3 : operator(discretization, kbins) => hyperparameter(kbins, strategy, choice).
h4 : operator(discretization, binarizer) => hyperparameter(binarizer, threshold, choice).
h5 : operator(normalization, robust_scaler) => hyperparameter(robust_scaler, with_centering, choice).
h6 : operator(normalization, robust_scaler) => hyperparameter(robust_scaler, with_scaling, choice).
h7 : operator(normalization, standard) => hyperparameter(standard, with_mean, choice).
h8 : operator(normalization, standard) => hyperparameter(standard, with_std, choice).

hc1 : operator(classification, dt) => hyperparameter(dt, max_depth, randint).
hc2 : operator(classification, dt) => hyperparameter(dt, min_samples_split, randint).
hc3 : operator(classification, dt) => hyperparameter(dt, min_samples_leaf, randint).
hc4 : operator(classification, dt) => hyperparameter(dt, max_features, randint).
hc5 : operator(classification, dt) => hyperparameter(dt, max_leaf_nodes, randint).
hc6 : operator(classification, dt) => hyperparameter(dt, splitter, choice).
hc7 : operator(classification, dt) => hyperparameter(dt, criterion, choice).
hc8 : operator(classification, knn) => hyperparameter(knn, n_neighbors, randint).
hc9 : operator(classification, knn) => hyperparameter(knn, weights, choice).
hc10 : operator(classification, knn) => hyperparameter(knn, metric, choice).

d1 : hyperparameter(kbins, n_bins, randint) => domain(kbins, n_bins, [3, 7]).
d2 : hyperparameter(kbins, encode, choice) => domain(kbins, encode, ["onehot", "onehot-dense", "ordinal"]).
d3 : hyperparameter(kbins, strategy, choice) => domain(kbins, strategy, ["uniform", "quantile", "kmeans"]).
d4 : hyperparameter(binarizer, threshold, choice) => domain(binarizer, threshold, [0.0, 0.5, 2.0, 5.0]).
d5 : hyperparameter(robust_scaler, with_centering, choice) => domain(robust_scaler, with_centering, [true, false]).
d6 : hyperparameter(robust_scaler, with_scaling, choice) => domain(robust_scaler, with_scaling, [true, false]).
d7 : hyperparameter(standard, with_mean, choice) => domain(standard, with_mean, [true, false]).
d8 : hyperparameter(standard, with_std, choice) => domain(standard, with_std, [true, false]).

dc1 : hyperparameter(dt, max_depth, randint) => domain(dt, max_depth, [1, 4]).
dc2 : hyperparameter(dt, min_samples_split, randint) => domain(dt, min_samples_split, [2, 5]).
dc3 : hyperparameter(dt, min_samples_leaf, randint) => domain(dt, min_samples_leaf, [1, 5]).
dc4 : hyperparameter(dt, max_features, randint) => domain(dt, max_features, [1, 3]).
dc5 : hyperparameter(dt, max_leaf_nodes, randint) => domain(dt, max_leaf_nodes, [2, 5]).
dc6 : hyperparameter(dt, splitter, choice) => domain(dt, splitter, ["best", "random"]).
dc7 : hyperparameter(dt, criterion, choice) => domain(dt, criterion, ["gini", "entropy"]).
dc8 : hyperparameter(knn, n_neighbors, randint) => domain(knn, n_neighbors, [3, 19]).
dc9 : hyperparameter(knn, weights, choice) => domain(knn, weights, ["uniform", "distance"]).
dc10 : hyperparameter(knn, metric, choice) => domain(knn, metric, ["minkowski", "euclidean", "manhattan"]).


c1 :=> mandatory_order([discretization, normalization], classification).
