s1 : [] => step(discretization).

o1 : step(discretization) => operator(discretization, kbins).
h1 : operator(discretization, kbins) => hyperparameter(kbins, n_bins, randint).
d1 : hyperparameter(kbins, n_bins, randint) => domain(kbins, n_bins, [2, 10]).


s2 : [] => step(normalization).
o3 : step(normalization) => operator(normalization, minmax).


s3 : [] => step(classification).

o4 : step(classification) => operator(classification, dt).
h4 : operator(classification, dt) => hyperparameter(dt, max_depth, randint).
d4 : hyperparameter(dt, max_depth, randint) => domain(dt, max_depth, [2, 30]).


c1 : [] => mandatory([discretization], dt).
