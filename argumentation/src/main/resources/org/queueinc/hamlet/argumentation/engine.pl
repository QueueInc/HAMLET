%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIPELINE GENERATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g0 : operator(classification, Z) => pipeline([], Z).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MANDATORY & FORBIDDEN CONSTRAINT
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

conflict([forbidden(SF, F, Y)], [mandatory(SM, M, Y)], check_conflict(SF, F, SM, M)).
conflict([mandatory(SM, M, Y)], [forbidden(SF, F, Y)], check_conflict(SF, F, SM, M)).

conflict([forbidden(Steps, Operators, Algorithm)], [pipeline(Operators2, Algorithm)], forbidden_conflict(Operators, Operators2)).

forbidden_conflict(Operators, Operators2) :-
	\+ (
		member(OperatorChoice, Operators),
		\+ (
			member(Operator, OperatorChoice),
			member(Operator, Operators2)
		)
	).

conflict([mandatory(Steps, Operators, Algorithm)], [pipeline(Operators2, Algorithm)], mandatory_conflict(Operators, Operators2)).

mandatory_conflict(Operators, Operators2) :-
	member(OperatorChoice, Operators),
	\+ (
		member(Operator, OperatorChoice),
		member(Operator, Operators2)
	).

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


len([], 0).
len([H|T], Z) :- len(T, Z1), Z is Z1 + 1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MANDATORY ORDER CONSTRAINT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cco0 : mandatory_order([A, B], Y), mandatory_order([B, C], Y), prolog(Y \= classification) => mandatory_order([A, C], Y).
cco1 : mandatory_order(X, classification), operator(classification, Y) => mandatory_order(X, Y).
cc02 : mandatory_order(X, Z, Y), prolog(Y \= classification), prolog(
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
	prolog((len(ZZ, L1), len(X, L2), L1 = L2)) => mandatory_order(X, ZZ, Y).


conflict([mandatory_order([A, C], Y)], [mandatory_order([C, A], Y)]).
conflict([mandatory_order(Steps, Operators, Algorithm)], [pipeline(Operators2, Algorithm)], mandatory_order_conflict(Operators, Operators2)).

mandatory_order_conflict([A, B], Pipeline) :-
	member(BB, B),
	member(AA, A),
	is_before(BB, AA, Pipeline), !.

is_before(A, B, [A|Tail]) :- member(B, Tail), !.
is_before(A, B, [_|Tail]) :- is_before(A, B, Tail).
