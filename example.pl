s1 : [] => step(discretization).
o1 : step(discretization) => operator(discretization, kbins).
h1 : operator(discretization, kbins) => hyperparameter(kbins, n_bins, randint).
d1 : hyperparameter(kbins, n_bins, randint) => domain(kbins, n_bins, [2, 6]).

s2 : [] => step(normalization).
o2 : step(normalization) => operator(normalization, standard).
h2 : operator(normalization, standard) => hyperparameter(standard, with_mean, choice).
d2 : hyperparameter(standard, with_mean, choice) => domain(standard, with_mean, [true, false]).
h3 : operator(normalization, standard) => hyperparameter(standard, with_std, choice).
d3 : hyperparameter(standard, with_std, choice) => domain(standard, with_std, [true, false]).

s3 : [] => step(classification).

o3 : step(classification) => operator(classification, dt).
h4 : operator(classification, dt) => hyperparameter(dt, max_depth, randint).
d4 : hyperparameter(dt, max_depth, randint) => domain(dt, max_depth, [1, 5]).

o4 : step(classification) => operator(classification, knn).
h5 : operator(classification, knn) => hyperparameter(knn, k, randint).
d5 : hyperparameter(knn, k, randint) => domain(knn, k, [6, 10]).

c1 : [] => mandatory([normalization], classification).
c2 : [] => forbidden([normalization], dt).
% c3 :=> hyperparameter_exception(classification, dt, max_depth, eq, 1).
% c4 :=> hyperparameter_exception(classification, dt, max_depth, gt, 2, [normalization]).
