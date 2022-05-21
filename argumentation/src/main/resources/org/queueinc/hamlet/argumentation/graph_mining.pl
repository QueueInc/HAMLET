fetch_complete_space([(prototype, choice, SPrototypes)|FSpace]) :-
    fetch_space(Space),
    map_space(Space, FSpace),
    fetch_prototypes(DPrototypes),
    findall(XX, (member(X, DPrototypes), concat(X, XX)), SPrototypes).

concat([H], H).
concat([H|T], HTT) :-
    concat(T, TT),
    atom_concat(H, '_', H1),
    atom_concat(H1, TT, HTT).

fetch_prototypes(DPrototypes) :-
   findall(Prototype, fetch_prototype(Prototype), Prototypes),
   distinct(Prototypes, [H|TPrototypes]),
   take_max(TPrototypes, [H], DPrototypes).

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

fetch_space(Space) :-
    findall(
        (Step, choice, OpWithHyperparameter),
        fetch_step_domain(Step, OpWithHyperparameter),
        Space
    ).

map_space([], []).
map_space([(classification, X, Y)|T], [(classification, X, Y)|TT]) :- !, map_space(T, TT).
map_space([(Z, X, Y)|T], [(Z, X, [(functionTransformer)|Y])|TT]) :- map_space(T, TT).


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

fetch_step(Step) :-
    get_argument_by_conclusion(step(Step)).

fetch_operator(Step, Operator) :-
    get_argument_by_conclusion(operator(Step, Operator)).

fetch_hyperparameter_with_domain(Operator, Hyperparameter, Type, Value) :-
    fetch_hyperparameter(Operator, Hyperparameter, Type),
    fetch_domain(Operator, Hyperparameter, Value).

fetch_hyperparameter(Operator, Hyperparameter, Type) :-
    get_argument_by_conclusion(hyperparameter(Operator, Hyperparameter, Type)).

fetch_domain(Operator, Hyperparameter, Value) :-
    get_argument_by_conclusion(domain(Operator, Hyperparameter, Value)).

fetch_prototype(Prototype) :-
    get_argument_by_conclusion(pipeline(Operators, Algorithm)),
    operatorToStep(Operators, Steps),
    append(Steps, [classification], Prototype).

operatorToStep([], []).
operatorToStep([H|T], [S|T1]) :-
    operatorToStep(T, T1),
    get_argument_by_conclusion(operator(S, H)).


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


distinct([], []).
distinct([H|T], T1) :- distinct(T, T1), member(H, T1), !.
distinct([H|T], [H|T1]) :- distinct(T, T1), \+ member(H, T1).


fetch_mandatory(Mandatory) :-
    findall((S, O, A), get_in_argument_by_conclusion(mandatory(S, O, A)), Mandatory).

fetch_forbidden(Forbidden) :-
    findall((S, O, A), get_in_argument_by_conclusion(forbidden(S, O, A)), Forbidden).


get_hyperparameters(Operator, Hyperparameters) :-
    findall(
        (Hyperparameter, Type, Value),
        (
            fetch_hyperparameter_with_domain(Operator, Hyperparameter, Type, Value)
        ),
        Hyperparameters
    ).

fetch_out_pipeline_support([], []).
fetch_out_pipeline_support([H|T], [(H, Hyperparameters)|R]) :-
    get_hyperparameters(H, Hyperparameters),
    fetch_out_pipeline_support(T, R).

fetch_out_pipeline(Pipeline) :-
    get_out_argument_by_conclusion(pipeline(Operators, Algorithm)),
    append(Operators, [Algorithm], T),
    fetch_out_pipeline_support(T, Pipeline).

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

pipeline_to_instance([], []).
pipeline_to_instance([(Step, (Operator, Hyperparameters))|T], [(Step, (Operator, Samples))|R]) :-
    pipeline_to_instance(T, R),
    sample_hyperparameters(Hyperparameters, Samples).


pipeline_to_prototype([], [], []).
pipeline_to_prototype([(Operator, Hyper)|T], [(Step, (Operator, Hyper))|R1], [Step|R2]) :-
    once(get_argument_by_conclusion(operator(Step, Operator))),
    pipeline_to_prototype(T, R1, R2).

missing_steps(Steps, DMissingSteps) :-
    findall(
        (Step, (functionTransformer)),
        (get_argument_by_conclusion(step(Step)), \+ member(Step, Steps)),
        MissingSteps
    ),
    distinct(MissingSteps, DMissingSteps).

fetch_out_instances(Instances) :-
    findall(
        [(prototype, Prototype)|InstanceWithMissingSteps],
        (
            fetch_out_pipeline(Pipeline),
            pipeline_to_prototype(Pipeline, PipelineWithSteps, Steps),
            missing_steps(Steps, MissingSteps),
            pipeline_to_instance(PipelineWithSteps, Instance),
            append(Instance, MissingSteps, InstanceWithMissingSteps),
            write(MissingSteps),nl,
            complete_prototype(Steps, MissingSteps, CompletePrototype),
            write(merda),nl,
            concat(CompletePrototype, Prototype)
        ),
        Instances
    ).

complete_prototype(Steps, Missing, Prototype) :-
    split_last(Steps, Transformations, Algorithm),
    clean_missing(Missing, CMissing),
    merge_with_permutations(Transformations, CMissing, Result),
    append(Result, [Algorithm], Prototype).

merge_with_permutations([], [], []).
merge_with_permutations(Target, Missing, [X|NR]) :-
    member(X, Missing),
    utils::subtract(Missing, [X], NMissing),
    merge_with_permutations(Target, NMissing, NR).
merge_with_permutations([X|NTarget], Missing, [X|NR]) :-
    merge_with_permutations(NTarget, Missing, NR).

split_last([H], [], H) :- !.
split_last([H|T], [H|TT], R) :-
    split_last(T, TT, R).

clean_missing([], []).
clean_missing([(H, C)|T], [H|R]) :- clean_missing(T, R).

% Attacca -> pipeline_instance(Operators, Algorithm, Hyperparameter, Comparator, Value)
% hyperparameter_exception(algorithm, dt, max_depth, eq, 1).

% Attacca -> pipeline_instance(Operators, Algorithm, Hyperparameter, Comparator, Value)
% hyperparameter_exception(algorithm, dt, max_depth, gt, 2, [normalization]).


dump_graph_data(Data) :-
    findall(argument(X), context_check(argument(X)), Arguments),
    findall(support(X, Y), context_check(support(X, Y)), Supports),
    findall(attack(X, Y, Z, Q), context_check(attack(X, Y, Z, Q)), Attacks),
    findall(in(X), context_check(in(X)), In),
    findall(out(X), context_check(out(X)), Out),
    findall(und(X), context_check(und(X)), Und),
    utils::appendLists([Arguments, Supports, Attacks, In, Out, Und], Data).

load_graph_data(Data) :-
    findall(_, (member(X, Data), context_assert(X)), _).
