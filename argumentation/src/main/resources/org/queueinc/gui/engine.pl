%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The following part is the engine that allows the argumation to be performed.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cc0 : mandatory(X, classification), operator(classification, Y) => mandatory(X, Y).
cc1 : forbidden(X, classification), operator(classification, Y) => forbidden(X, Y).

cc2 : mandatory(X, Y), prolog(Y \= classification), prolog(
		findall(Z,
			(
				member(XX, X),
				\+ compound(XX),
				findall(no, \+ context_check(argument([_, _, [operator(XX, XXX)], _, _])), T),
				T = [],
				findall(XXX, context_check(argument([_, _, [operator(XX, XXX)], _, _])), Z)
			),
			ZZ
		)
	),
	prolog((len(ZZ, L1), len(X, L2), L1 = L2)) => mandatory(X, ZZ, Y).

cc3 : forbidden(X, Y), prolog(Y \= classification), prolog(
		findall(Z ,
			(
				member(XX, X),
				\+ compound(XX),
				findall(no, \+ context_check(argument([_, _, [operator(XX, XXX)], _, _])), T),
				T = [],
				findall(XXX, context_check(argument([_, _, [operator(XX, XXX)], _, _])), Z)
			),
			ZZ
		)
	),
	prolog((len(ZZ, L1), len(X, L2), L1 = L2)) => forbidden(X, ZZ, Y).

len([], 0).
len([H|T], Z) :- len(T, Z1), Z is Z1 + 1.

conflict([forbidden(SF, F, Y)], [mandatory(SM, M, Y)]) :- check_conflict(SF, F, SM, M).
conflict([mandatory(SM, M, Y)], [forbidden(SF, F, Y)]) :- check_conflict(SF, F, SM, M).

b([S|_], [H4|_], S, OP) :- subset(OP, H4), !.
b([H3|T3], [_|T4], S, OP) :- H3 \= S, b(T3, T4, S, OP).

a([], [], _, _).
a([H1|T1], [H2|T2], SF, F) :- b(SF, F, H1, H2), a(T1, T2, SF, F).

subset([], _).
subset([H|T], Y) :- member(H, Y), subset(T, Y).


get_subsets([], [], _, [], []).
get_subsets([H1|T1], [H2|T2], SM, [H1|NSF], [H2|NF]) :-
	member(H1, SM),
	get_subsets(T1, T2, SM, NSF, NF).
get_subsets([H1|T1], [_|T2], SM, NSF, NF) :-
	\+ member(H1, SM),
	get_subsets(T1, T2, SM, NSF, NF).

check_conflict(SF, F, SM, M) :-
	subset(SF, SM),
	get_subsets(SM, M, SF, NSM, NM),
	a(NSM, NM, SF, F).

% g2 : step(X), step(Y), step(classification), prolog((X \= Y, X \= classification, Y \= classification)) => pipeline_prototype([X, Y], classification).
% g3 : step(X), step(classification), prolog(X \= classification) => pipeline_prototype([X], classification).

g4 : operator(classification, Z) => pipeline([], Z).
g5 : step(X), step(classification), prolog(X \= classification), operator(X, XX), operator(classification, ZZ) => pipeline([XX], ZZ).
g6 : step(X), step(Y), step(classification), prolog((X \= Y, X \= classification, Y \= classification)),
		operator(X, XX), operator(Y, YY), operator(classification, ZZ) => pipeline([XX, YY], ZZ).

% controllo sull'errore
g7 : prolog(context_check(argument([_, _, [hyperparameter_exception(classification, OP, H, COMP, VALUE)], _, _]))),
	pipeline(X, OP),
	hyperparameter(OP, H, DOMAIN_TYPE),
	~(hyperparameter_exception(classification, OP, H, COMP, VALUE)) => pipeline_instance(X, OP, H, COMP, VALUE).

g8 : prolog(context_check(argument([_, _, [hyperparameter_exception(P, OP, H, COMP, VALUE)], _, _]))),
	pipeline(X, OP),
	prolog(member(H, X)),
	hyperparameter(OP, H, DOMAIN_TYPE),
	~(hyperparameter_exception(P, OP, H, COMP, VALUE)) => pipeline_instance(X, OP, H, COMP, VALUE).

g9 : prolog(context_check(argument([_, _, [hyperparameter_exception(classification, OP, H, COMP, VALUE, EXCEPTION)], _, _]))),
	pipeline(X, OP),
	hyperparameter(OP, H, DOMAIN_TYPE),
	prolog(member(XX, X)),
	operator(Y, XX),
	prolog(member(Y, EXCEPTION)),
	~(hyperparameter_exception(classification, OP, H, COMP, VALUE, EXCEPTION)) => pipeline_instance(X, OP, H, COMP, VALUE).

g10 : prolog(context_check(argument([_, _, [hyperparameter_exception(P, OP, H, COMP, VALUE, EXCEPTION)], _, _]))),
	pipeline(X, OP),
	prolog(member(H, X)),
	hyperparameter(OP, H, DOMAIN_TYPE),
	prolog(member(XX, X)),
	operator(Y, XX),
	prolog(member(Y, EXCEPTION)),
	~(hyperparameter_exception(P, OP, H, COMP, VALUE, EXCEPTION)) => pipeline_instance(X, OP, H, COMP, VALUE).


conflict([forbidden(Steps, Operators, Algorithm)], [pipeline(Operators2, Algorithm)]) :-
	\+ (
		member(OperatorChoice, Operators),
		\+ (
			member(Operator, OperatorChoice),
			member(Operator, Operators2)
		)
	).

conflict([mandatory(Steps, Operators, Algorithm)], [pipeline(Operators2, Algorithm)]) :-
	member(OperatorChoice, Operators),
	\+ (
		member(Operator, OperatorChoice),
		member(Operator, Operators2)
	).
