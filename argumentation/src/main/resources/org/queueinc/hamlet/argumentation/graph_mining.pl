%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DATA IMPORT/EXPORT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dump_graph_data(Data) :-
    findall(argument(X), context_check(argument(X)), Arguments),
    findall(support(X, Y), context_check(support(X, Y)), Supports),
    findall(attack(X, Y, Z, Q), context_check(attack(X, Y, Z, Q)), Attacks),
    findall(in(X), context_check(in(X)), In),
    findall(out(X), context_check(out(X)), Out),
    findall(und(X), context_check(und(X)), Und),
    utils::appendLists([Arguments, Supports, Attacks, In, Out, Und], Data).


load_graph_data(Data) :- utils::assert_all(Data).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% UTILS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

len([], 0).
len([H|T], Z) :- len(T, Z1), Z is Z1 + 1.


take_max([], X, X).
take_max([H|T], LISTMAX, R) :-
    member(MAX, LISTMAX),
    len(H, LH),
    len(MAX, LMAX),
    LH > LMAX,
    take_max(T, [H], R), !.
take_max([H|T], LISTMAX, R) :-
    member(MAX, LISTMAX),
    len(H, LH),
    len(MAX, LMAX),
    LH =:= LMAX,
    take_max(T, [H|LISTMAX], R), !.
take_max([_|T], LISTMAX, R) :-
    take_max(T, LISTMAX, R).


split_last([H], [], H) :- !.
split_last([H|T], [H|TT], R) :-
    split_last(T, TT, R).


concat([H], H).
concat([H|T], HTT) :-
    concat(T, TT),
    atom_concat(H, '_', H1),
    atom_concat(H1, TT, HTT).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DATA LAYER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


get_in_argument_by_conclusion(Conc) :-
    context_check(in([_, _, [Conc], _, _])).


get_out_argument_by_conclusion(Conc) :-
    context_check(out([_, _, [Conc], _, _])).


get_argument_by_conclusion(Conc) :-
    context_check(argument([_, _, [Conc], _, _])).


get_argument_by_conclusion(Conc, [A, B, [Conc], C, D]) :-
    context_check(argument([A, B, [Conc], C, D])).


get_attacked_by_conclusion(Conc, Attacker) :-
    context_check(attack(_, Attacker, [_, _, [Conc], _, _], _)).


fetch_step(Step) :-
    get_argument_by_conclusion(step(Step)).


fetch_operator(Step, Operator) :-
    get_argument_by_conclusion(operator(Step, Operator)).


fetch_operators([], []).
fetch_operators([H|T], [HH|TT]) :-
    findall(Op, fetch_operator(S, Op), HH),
    fetch_operators(T, TT).


get_hyperparameters(Operator, Hyperparameters) :-
    findall(
        (Hyperparameter, Type, Value),
        (
            fetch_hyperparameter_with_domain(Operator, Hyperparameter, Type, Value)
        ),
        Hyperparameters
    ).


fetch_hyperparameter_with_domain(Operator, Hyperparameter, Type, Value) :-
    fetch_hyperparameter(Operator, Hyperparameter, Type),
    fetch_domain(Operator, Hyperparameter, Value).


fetch_hyperparameter(Operator, Hyperparameter, Type) :-
    get_argument_by_conclusion(hyperparameter(Operator, Hyperparameter, Type)).


fetch_domain(Operator, Hyperparameter, Value) :-
    get_argument_by_conclusion(domain(Operator, Hyperparameter, Value)).


fetch_prototype(Prototype) :-
    get_argument_by_conclusion(pipeline(Steps, Algorithm)),
    append(Steps, [classification], Prototype).


fetch_prototypes(DPrototypes) :-
   findall(Prototype, fetch_prototype(Prototype), Prototypes),
   utils::deduplicate(Prototypes, [H|TPrototypes]),
   take_max(TPrototypes, [H], DPrototypes).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPACE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fetch_complete_space([(prototype, choice, SPrototypes)|FSpace]) :-
    fetch_space(Space),
    map_space(Space, FSpace),
    fetch_prototypes(DPrototypes),
    findall(XX, (member(X, DPrototypes), concat(X, XX)), SPrototypes).


fetch_space(Space) :-
    findall(
        (Step, choice, OpWithHyperparameter),
        fetch_step_domain(Step, OpWithHyperparameter),
        Space
    ).


fetch_step_domain(Step, OpWithHyperparameter) :-
    fetch_step(Step),
    findall(Operator, fetch_operator(Step, Operator), Operators),
    findall(
        (Op, Hyperparameters),
        (
            member(Op, Operators),
            findall(
                (Hyperparameter, Type, Value),
                (
                    fetch_hyperparameter(Op, Hyperparameter, Type),
                    fetch_domain(Op, Hyperparameter, Value)
                ),
                Hyperparameters
            )
        ),
        OpWithHyperparameter
    ).


map_space([], []).
map_space([(classification, X, Y)|T], [(classification, X, Y)|TT]) :- !, map_space(T, TT).
map_space([(Z, X, Y)|T], [(Z, X, [(function_transformer)|Y])|TT]) :- map_space(T, TT).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONSTRAINTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fetch_mandatory(Mandatory) :-
    findall((S, O, A), (
        get_in_argument_by_conclusion(mandatory(S, A)),
        fetch_operators(S, O)
    ), Mandatory).


fetch_forbidden(Forbidden) :-
    findall((S, O, A), (
        get_in_argument_by_conclusion(forbidden(S, A)),
        fetch_operators(S, O)
    ), Forbidden).


fetch_mandatory_order(Mandatory) :-
    findall((S, O, A), (
        get_in_argument_by_conclusion(mandatory_order(S, A)),
        fetch_operators(S, O)
    ), Mandatory).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUT INSTANCES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


generate_samples(A, B, 0, [A, B]) :- !.
generate_samples(A, B, Step, [B]) :-
    X is A + Step,
    X >= B, !.
generate_samples(A, B, Step, [X|R]) :-
    X is A + Step,
    X < B,
    generate_samples(X, B, Step, R).

sample_range(A, B, Sample) :-
    Step is (B - A) // 10,
    generate_samples(A, B, Step, R),
    member(Sample, R).

sample_hyperparameters([], []).
sample_hyperparameters([(Hyperparameter, choice, Value)|T], [(Hyperparameter, X)|R]) :-
    member(X, Value),
    sample_hyperparameters(T, R).
sample_hyperparameters([(Hyperparameter, randint, [A, B])|T], [(Hyperparameter, X)|R]) :-
    sample_range(A, B, X),
    sample_hyperparameters(T, R).


step_instances(OpWithHyperparameter) :-
    findall(
        (Step, Op, IHyperparameters),
        (
            fetch_operator(Step, Op),
            findall(
                (Hyperparameter, Type, Value),
                (
                    fetch_hyperparameter(Op, Hyperparameter, Type),
                    fetch_domain(Op, Hyperparameter, Value)
                ),
                Hyperparameters
            ),
	    sample_hyperparameters(Hyperparameters, IHyperparameters)
        ),
        OpWithHyperparameter
    ).


steps(S) :- findall(Step, (fetch_step(Step), Step \= classification), S).

merge_with_perm([], [], []).
merge_with_perm(Target, Missing, [(X, function_transformer)|NR]) :-
    member(X, Missing),
    utils::subtract(Missing, [X], NMissing),
    merge_with_perm(Target, NMissing, NR).
merge_with_perm([X|NTarget], Missing, [X|NR]) :-
    merge_with_perm(NTarget, Missing, NR).

out_prototype(AllSteps, P) :-
    get_out_argument_by_conclusion(pipeline(Steps, Algorithm)),
    utils::subtract(AllSteps, Steps, Missing),
    merge_with_perm(Steps, Missing, F),
    utils::append_fast(F, [(classification, Algorithm)], P).


merge_prototypes([], _, []).
merge_prototypes([(classification, X)|T], Ops, [(classification, X, Hyper)|TT]) :-
    member((classification, X, Hyper), Ops),
    merge_prototypes(T, Ops, TT).
merge_prototypes([(X, function_transformer)|T], Ops, [(X, function_transformer)|TT]) :-
    merge_prototypes(T, Ops, TT).
merge_prototypes([X|T], Ops, [(X, Op, Hyper)|TT]) :-
    \+ compound(X),
    member((X, Op, Hyper), Ops),
    merge_prototypes(T, Ops, TT).


fetch_out_instance_components(Ops, Ps) :-
    steps(AllSteps),
    findall(P, out_prototype(AllSteps, P), Ps),
    step_instances(Ops).

fetch_out_instance(R) :-
    steps(AllSteps),
    step_instances(Ops),
    out_prototype(AllSteps, P),
    merge_prototypes(P, Ops, R).

fetch_out_instances(P) :- findall(R, fetch_out_instance(R), P).
