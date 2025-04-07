%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FAIRNESS GROUPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prepare_sensitive_groups(Res) :-
     findall(Num :=> sensitive_group(X), (
        sensitive_group(X),
        rand_int(0, 1000000, Num)
     ), Res).

sensitive_group([Y]) :-
    sensitive_feature(X, V),
    member(Y, V).
sensitive_group(RR) :-
    findall(V, sensitive_feature(X, V), R),
    cartesian_product(R, RR).

cartesian_product([], []).
cartesian_product([H|T], [X|R]) :-
    member(X, H),
    cartesian_product(T, R).

% Exact Check
% conflict([sensitive_group(X)], [instance(Y, Z)], once(discriminate_check_order(instance(Y, Z), X))).
conflict([sensitive_group(X)], [pipeline(Y, Z)], discriminate_check_order(pipeline(Y, Z), X)).

discriminate_check(pipeline(Y, Z), X) :-
    permutations(X, PX),
    discriminate(pipeline(Y, Z), PX).
discriminate_check(instance(Y, Z), X) :-
    permutations(X, PX),
    discriminate(instance(Y, Z), PX).


discriminate_check_order(instance(Y, Z), X) :-
    permutations(X, PX),
    discriminate(instance(PY, Z), PX),
    same_order(PY, Y).
discriminate_check_order(pipeline(Y, Z), X) :-
    permutations(X, PX),
    discriminate(pipeline(PY, Z), PX),
    check_order(PY, Y).

check_order(PY, Y) :-
    var(Y), !,
    matching_prototypes(PY, [], Y).
check_order(PY, Y) :- same_order(PY, Y).

matching_prototypes([], X, X).
matching_prototypes([H|T], I, X) :- append(I, [H], II), matching_prototypes(T, II, X).
matching_prototypes(L, I, X) :- step(S), S \= classification, \+ member(S, I), append(I, [S], II), matching_prototypes(L, II, X).


permutations([], []).
permutations(H, [X|O]) :-
    member(X, H),
    utils::subtract(H, [X], R),
    permutations(R, O).


same_order([], _).
same_order([Y|T], [Y|PT]) :- same_order(T, PT).
same_order([Y|T], [_|PT]) :- same_order([Y|T], PT).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PIPELINE GENERATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pipeline([], Z) :- operator(classification, Z).

prepare_theory(Res) :-
    once(constraints(CS)),
    prepare_pipelines(CS, PS),
    prepare_pipelines_instances(CS, PS, IS),
    utils::append_fast(IS, PS, Res).

prepare_pipelines(CS, Res) :-
	findall(Num :=> pipeline(X, Y), (
	    pipeline(X, Y),
	    % once(check_potential_conflict(CS, pipeline(X, Y))),
	    check_potential_conflict(CS, pipeline(X, Y)),
	    rand_int(0, 1000000, Num)
    ), Res).

prepare_pipelines_instances(CS, PS, Res) :-
	findall(Num : pipeline(P, Y)  => instance(Z, Y), (
	    check_potential_conflict(CS, instance(Z, Y)),
	    build_pipeline(Z, P),
	    rand_int(0, 1000000, Num)
    ), Instances),
    check_pipelines(Instances, PS, NP),
    utils::appendLists([Instances, NP], Res).

check_pipelines(I, P, NP) :-
    once(setof(X, (member(_ : X  => _, I), \+ member(_ :=> X, P)), R); R = []),
    findall(Num :=> X, (member(X, R), rand_int(0, 1000000, Num)), NP).

build_pipeline([], []).
build_pipeline([H|T], [O|R]) :-
    operator(O, H),
    build_pipeline(T, R).

build_instance([], []).
build_instance([H|T], [O|R]) :-
    operator(H, O),
    build_instance(T, R).

check_potential_conflict([C|_], R) :-
    expanded_conflict(C, [R]).
check_potential_conflict([_|T], R) :-
    check_potential_conflict(T, R).

expanded_conflict(HeadA, HeadB) :-
    conflict(HeadA, HeadB).
expanded_conflict(HeadA, HeadB) :-
    conflict(HeadA, HeadB, Guard),
    (callable(Guard) -> call(Guard); Guard).

constraints(XS) :-
    context_active(OC),
    context_branch(OC, _),
    buildLabelSetsSilent,
    findall(X, context_check(clause(conc(X), argument(_))), XS),
    context_checkout(OC).

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
