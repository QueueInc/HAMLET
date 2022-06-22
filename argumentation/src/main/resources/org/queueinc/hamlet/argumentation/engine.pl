%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIPELINE GENERATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pipeline([], Z) :- operator(classification, Z).

preparePipelines(Res) :-
	findall(Num :=> pipeline(X, Y), (pipeline(X, Y), rand_int(0, 1000000, Num)), Res).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MANDATORY & FORBIDDEN CONSTRAINT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cc0 : mandatory(X, classification), prolog(operator(classification, Y)) => mandatory(X, Y).
cc1 : forbidden(X, classification), prolog(operator(classification, Y)) => forbidden(X, Y).

conflict([forbidden(SF, Y)], [mandatory(SM, Y)], once(subset(SF, SM))).
conflict([mandatory(SM, Y)], [forbidden(SF, Y)], once(subset(SF, SM))).

conflict([forbidden(Steps, Algorithm)], [pipeline(Steps2, Algorithm)], once(forbidden_conflict(Steps, Steps2))).

forbidden_conflict(Steps, Steps2) :-
	\+ (
		member(Step, Steps),
		\+ member(Step, Steps2)
	).

conflict([mandatory(Steps, Algorithm)], [pipeline(Steps2, Algorithm2)], once(mandatory_conflict(Steps, Steps2, Algorithm, Algorithm2))).

mandatory_conflict([], _, Algorithm, Algorithm2) :- Algorithm \= Algorithm2.
mandatory_conflict(Steps, Steps2, Algorithm, Algorithm) :-
	member(Step, Steps),
	\+ member(Step, Steps2).

subset([], _).
subset([H|T], Y) :- member(H, Y), subset(T, Y).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MANDATORY ORDER CONSTRAINT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cco0 : mandatory_order(X, classification), prolog(operator(classification, Y)) => mandatory_order(X, Y).

conflict([mandatory_order([A, C], Y)], [mandatory_order([C, A], Y)]).
conflict([mandatory_order(Steps, Algorithm)], [pipeline(Steps2, Algorithm)], once(mandatory_order_conflict(Steps, Steps2))).

mandatory_order_conflict([A, B], Pipeline) :-
	is_before(B, A, Pipeline), !.

is_before(A, B, [A|Tail]) :- member(B, Tail), !.
is_before(A, B, [_|Tail]) :- is_before(A, B, Tail).
