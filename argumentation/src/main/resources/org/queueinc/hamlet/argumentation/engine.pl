%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIPELINE GENERATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g0 : operator(classification, Z) => pipeline([], Z).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MANDATORY & FORBIDDEN CONSTRAINT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cc0 : mandatory(X, classification), operator(classification, Y) => mandatory(X, Y).
cc1 : forbidden(X, classification), operator(classification, Y) => forbidden(X, Y).

conflict([forbidden(SF, Y)], [mandatory(SM, Y)], subset(SF, SM)).
conflict([mandatory(SM, Y)], [forbidden(SF, Y)], subset(SF, SM)).

conflict([forbidden(Steps, Algorithm)], [pipeline(Steps2, Algorithm)], forbidden_conflict(Steps, Steps2)).

forbidden_conflict(Steps, Steps2) :-
	\+ (
		member(Step, Steps),
		\+ member(Step, Steps2)
	).

conflict([mandatory(Steps, Algorithm)], [pipeline(Steps2, Algorithm2)], mandatory_conflict(Steps, Steps2, Algorithm, Algorithm2)).

mandatory_conflict([], _, Algorithm, Algorithm2) :- Algorithm \= Algorithm2.
mandatory_conflict(Steps, Steps2, Algorithm, Algorithm) :-
	member(Step, Steps),
	\+ member(Step, Steps2).

subset([], _).
subset([H|T], Y) :- member(H, Y), subset(T, Y).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MANDATORY ORDER CONSTRAINT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cco0 : mandatory_order(X, classification), operator(classification, Y) => mandatory_order(X, Y).

conflict([mandatory_order([A, C], Y)], [mandatory_order([C, A], Y)]).
conflict([mandatory_order(Steps, Algorithm)], [pipeline(Steps2, Algorithm)], mandatory_order_conflict(Steps, Steps2)).

mandatory_order_conflict([A, B], Pipeline) :-
	is_before(B, A, Pipeline), !.

is_before(A, B, [A|Tail]) :- member(B, Tail), !.
is_before(A, B, [_|Tail]) :- is_before(A, B, Tail).
