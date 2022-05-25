s1 :=> step(discretization).

o1 : step(discretization) => operator(discretization, kbins).
h1 : operator(discretization, kbins) => hyperparameter(kbins, n_bins, randint).
d1 : hyperparameter(kbins, n_bins, randint) => domain(kbins, n_bins, [2, 10]).


s2 :=> step(normalization).

o2 : step(normalization) => operator(normalization, standard).
h2 : operator(normalization, standard) => hyperparameter(standard, with_mean, choice).
d2 : hyperparameter(standard, with_mean, choice) => domain(standard, with_mean, [true, false]).
h3 : operator(normalization, standard) => hyperparameter(standard, with_std, choice).
d3 : hyperparameter(standard, with_std, choice) => domain(standard, with_std, [true, false]).

o3 : step(normalization) => operator(normalization, minmax).

s3 :=> step(classification).

o4 : step(classification) => operator(classification, dt).
h4 : operator(classification, dt) => hyperparameter(dt, max_depth, randint).
d4 : hyperparameter(dt, max_depth, randint) => domain(dt, max_depth, [2, 30]).

o4 : step(classification) => operator(classification, knn).
h5 : operator(classification, knn) => hyperparameter(knn, n_neighbors, randint).
d5 : hyperparameter(knn, n_neighbors, randint) => domain(knn, n_neighbors, [2, 30]).

s4 :=> step(features_engineering).
o5 : step(features_engineering) => operator(features_engineering, select_k_best).

c1 :=> mandatory([discretization], dt).
c2 :=> forbidden([normalization], dt).
c3 :=> mandatory([normalization], knn).

c4 :=> mandatory_order([discretization, normalization], classification).

% c3 :=> hyperparameter_exception(classification, dt, max_depth, eq, 1).
% c4 :=> hyperparameter_exception(classification, dt, max_depth, gt, 2, [normalization]).