s1 : [] => step(discretization).
o1 : step(discretization) => operator(discretization, kbins).
h1 : operator(discretization, kbins) => hyperparameter(kbins, n_bins, randint).
d1 : hyperparameter(kbins, n_bins, randint) => domain(kbins, n_bins, 2, 6).

s2 : [] => step(normalization).
o2 : step(normalization) => operator(normalization, standard).
h2 : operator(normalization, standard) => hyperparameter(standard, with_mean, choice).
d2 : hyperparameter(standard, with_mean, choice) => domain(with_mean, choice, [true, false]).
h3 : operator(normalization, standard) => hyperparameter(standard, with_std, choice).
d3 : hyperparameter(standard, with_std, choice) => domain(with_std, choice, [true, false]).

s3 : [] => step(algorithm).
o3 : step(algorithm) => operator(algorithm, dt).
h4 : operator(algorithm, dt) => hyperparameter(dt, max_depth, randint).
d4 : hyperparameter(dt, max_depth, randint) => domain(max_depth, randint, 1, 5).

% dominio iperparametro su operatore
% argomento per eccezione iperparametro su pipeline  

% hyperparameter => hyper_domain(5, 8)
% pipeline, hyper_domain => 
% if hyper x has value y & z in pipeline => ko
% if hyper x has value y then hyper z must have value p

c1 :=> mandatory_steps_for_algorithm([discretization], dt).
c2 :=> incompatible_steps([normalization, discretization]).
c3 :=> hyperparameter_exception(algorithm, dt, max_depth, eq, 1).
c4 :=> hyperparameter_exception(algorithm, dt, max_depth, gt, 2, [normalization]).

% The following part is the engine that allows the argumation to be performed.
% It will be hidden, since will be part of the HAMLET framework.

g1 : mandatory_steps_for_algorithm([X], Y), operator(X, XX) => mandatory_operator_for_algorithm([XX], Y).
g2 : step(X), step(Y), step(algorithm), prolog((X \= Y, X \= algorithm, Y \= algorithm)) => pipeline_prototype([X, Y], algorithm).
g3 : step(X), step(algorithm), prolog(X \= algorithm) => pipeline_prototype([X], algorithm).
g4 : operator(algorithm, Z) => pipeline([], Z).
g5 : pipeline_prototype([X], Z), operator(X, XX), operator(Z, ZZ) => pipeline([XX], ZZ).
g6 : pipeline_prototype([X, Y], Z), operator(X, XX), operator(Y, YY), operator(Z, ZZ) => pipeline([XX, YY], ZZ).

% pipeline(X, Y), prolog(member(XX, X)), operator(normalization, XX)

% direttiva per eccezioni
% pipeline sbagliata
% wrong_instance(X, Y, Z)

% g7 : prolog(wrong_instance(X, Y, Z)), ~(hyperparameter_exception(X, Y, Z)), pipeline(X, Y, Z) => pipeline_instance(X, Y).

% controllo sull'errore
g7 : prolog(context_check(argument([_, _, [hyperparameter_exception(algorithm, OP, H, COMP, VALUE)], _, _]))),
	pipeline(X, OP),
	hyperparameter(OP, H, DOMAIN_TYPE),
	~(hyperparameter_exception(algorithm, OP, H, COMP, VALUE)) => pipeline_instance(X, OP, H, COMP, VALUE).

g8 : prolog(context_check(argument([_, _, [hyperparameter_exception(P, OP, H, COMP, VALUE)], _, _]))),
	pipeline(X, OP),
	prolog(member(H, X)),
	hyperparameter(OP, H, DOMAIN_TYPE),
	~(hyperparameter_exception(P, OP, H, COMP, VALUE)) => pipeline_instance(X, OP, H, COMP, VALUE).
	
g9 : prolog(context_check(argument([_, _, [hyperparameter_exception(algorithm, OP, H, COMP, VALUE, EXCEPTION)], _, _]))),
	pipeline(X, OP),
	hyperparameter(OP, H, DOMAIN_TYPE),
	prolog(member(XX, X)),
	operator(Y, XX),
	prolog(member(Y, EXCEPTION)),
	~(hyperparameter_exception(algorithm, OP, H, COMP, VALUE, EXCEPTION)) => pipeline_instance(X, OP, H, COMP, VALUE).

g10 : prolog(context_check(argument([_, _, [hyperparameter_exception(P, OP, H, COMP, VALUE, EXCEPTION)], _, _]))),
	pipeline(X, OP),
	prolog(member(H, X)),
	hyperparameter(OP, H, DOMAIN_TYPE),
	prolog(member(XX, X)),
	operator(Y, XX),
	prolog(member(Y, EXCEPTION)),
	~(hyperparameter_exception(P, OP, H, COMP, VALUE, EXCEPTION)) => pipeline_instance(X, OP, H, COMP, VALUE).

conflict([incompatible_steps(X)], [pipeline_prototype(Z, _)]) :- \+ (member(Y, X), \+ member(Y, Z)).
conflict([mandatory_operator_for_algorithm(X, Y)], [pipeline(Z, Y)]) :- member(T, X), \+ member(T, Z).

