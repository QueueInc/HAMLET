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
    take_max_support(LH, LMAX, H, LISTMAX, RES),
    take_max(T, RES, R), !.

take_max_support(L, A, H, LIST, [H]) :- L > A, !.
take_max_support(L, A, H, LIST, [H|LIST]) :- L =:= A, !.
take_max_support(_, _, _, LIST, LIST).


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

get_out_und_argument_by_conclusion(Conc) :-
    get_und_argument_by_conclusion(Conc).
get_out_und_argument_by_conclusion(Conc) :-
    get_out_argument_by_conclusion(Conc).


get_und_argument_by_conclusion(Conc) :-
    context_check(und([_, _, [Conc], _, _])).


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


fetch_step(Step) :- step(Step).


fetch_operator(Step, Operator) :- operator(Step, Operator).


fetch_operators([], []).
fetch_operators([H|T], [HH|TT]) :-
    findall(Op, fetch_operator(H, Op), HH),
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
    hyperparameter(Operator, Hyperparameter, Type).


fetch_domain(Operator, Hyperparameter, Value) :-
    domain(Operator, Hyperparameter, Value).


%fetch_prototype(Prototype) :-
%    get_argument_by_conclusion(pipeline(Steps, Algorithm)),
%    append(Steps, [classification], Prototype).


%fetch_prototypes(RPrototypes) :-
%    setof(Steps, get_argument_by_conclusion(pipeline(Steps, Algorithm)), [H|Prototypes]),
%    take_max(Prototypes, [H], DPrototypes),
%    findall(X, (member(Y, DPrototypes), append(Y, [classification], X)), RPrototypes).

fetch_prototypes(Rprototypes) :-
    findall(X, (step(X), X \= classification), Res),
    findall(Prot, (perm(Res, Pres), append(Pres, [classification], Prot)), Rprototypes).

takeout(X,[X|R],R).
takeout(X,[F |R],[F|S]) :- takeout(X,R,S).

perm([X|Y],Z) :- perm(Y,W), takeout(X,Z,W).
perm([],[]).


fetch_sensitive_features(X) :-
    findall(Y, sensitive_feature(Y, _), X).


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

map_to(_, [], []).
map_to(M, [_|T], [M|R]) :- map_to(M, T, R).

fetch_mandatory(Mandatory) :-
    findall((S, R, O, A), (
        get_in_argument_by_conclusion(mandatory(S, A)),
        A \= classification,
        fetch_operators(S, O),
        map_to(nin, S, R)
    ), Mandatory).


fetch_forbidden(Forbidden) :-
    findall((S, R, O, A), (
        get_in_argument_by_conclusion(forbidden(S, A)),
        A \= classification,
        fetch_operators(S, O),
        map_to(in, S, R)
    ), Forbidden).


fetch_mandatory_order(Mandatory) :-
    fetch_prototypes(Prototypes),
    findall(([prototype|S], [nin|R], [Ps|OPS], A), (
        get_in_argument_by_conclusion(mandatory_order(S, A)),
        A \= classification,
        findall(P, (
            member(P, Prototypes),
            match_prototype(P, S)
        ), Ps),
        map_to(neq, S, R),
        map_to(function_transformer, S, OPS)
    ), Mandatory).


fetch_forbidden_instances(Instances) :-
    fetch_prototypes(Prototypes),
    findall(([prototype|FS], [in|R], [Ps|FO], A), (
        discriminate(instance(O, A), _),
        A \= classification,
        build_pipeline(O, S),
        findall(P, (
            member(P, Prototypes),
            match_prototype(P, S)
        ), Ps),
        findall(S1, (step(S1), S1 \= classification, \+ member(S1, S)), NS),
        map_to(function_transformer, NS, NO),
        utils::append_fast(S, NS, FS),
        utils::append_fast(O, NO, FO),
        map_to(eq, FS, R)
    ), Instances).

fetch_forbidden_pipelines(Pipelines) :-
    fetch_prototypes(Prototypes),
    findall(([prototype|FS], [in|FR], [Ps|FO], A), (
        discriminate(pipeline(S, A), _),
        A \= classification,
        fetch_operators(S, O),
        findall(P, (
            member(P, Prototypes),
            match_prototype(P, S)
        ), Ps),
        findall(S1, (step(S1), S1 \= classification, \+ member(S1, S)), NS),
        map_to(function_transformer, NS, NO),
        utils::append_fast(S, NS, FS),
        utils::append_fast(O, NO, FO),
        map_to(eq, NS, NR),
        map_to(in, S, R),
        utils::append_fast(R, NR, FR)
    ), Pipelines).


match_prototype(_, []) :- !.
match_prototype([H|T], [H|TT]) :- !, match_prototype(T, TT).
match_prototype([_|T], TT) :- match_prototype(T, TT).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUT INSTANCES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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


valorize_operators(_, [], [], []).
valorize_operators(AllOperators, [(S, O)|T], [(S, O)|TT], [S|TTT]) :-
    valorize_operators(AllOperators, T, TT, TTT).
valorize_operators(AllOperators, [H|T], [(H, Operator)|TT], [H|TTT]) :-
    \+ compound(H),
    member((H, Operator), AllOperators),
    valorize_operators(AllOperators, T, TT, TTT).


fetch_instance_base([(prototype, CC)|R]) :-
    steps(AllSteps),
    findall((S,O), operator(S, O), AllOperators),
    out_prototype(AllSteps, P),
    valorize_operators(AllOperators, P, R, C),
    concat(C, CC).

fetch_all_instances_base(P) :- findall(R, fetch_instance_base(R), P).

fetch_instance_base_components(AllOperators, Ps) :-
    steps(AllSteps),
    findall(P, out_prototype(AllSteps, P), Ps),
    findall((S,O), operator(S, O), AllOperators).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPERATORS SAMPLING
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


fetch_out_instance(R) :-
    steps(AllSteps),
    step_instances(Ops),
    out_prototype(AllSteps, P),
    merge_prototypes(P, Ops, R).

fetch_out_instances(P) :- findall(R, fetch_out_instance(R), P).


fetch_out_instance_components(Ops, Ps) :-
    steps(AllSteps),
    findall(P, out_prototype(AllSteps, P), Ps),
    step_instances(Ops).
