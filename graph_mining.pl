fetch_prototypes(Prototypes) :-
   findall(Prototype, fetch_prototype(Prototype), Prototypes).

fetch_space(Space) :-
    findall(
	(Step, OpWithHyperparameter),
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
	    write(Op),nl,
            findall(
                (Hyperparameter, Type, Value),
                (
                    fetch_hyperparameter(Op, Hyperparameter, Type),
		    write(Hyperparameter),nl,
                    fetch_domain(Op, Hyperparameter, Value),
		    write(Value),nl
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

fetch_hyperparameter(Operator, Hyperparameter, Type) :-
    get_argument_by_conclusion(hyperparameter(Operator, Hyperparameter, Type)).

fetch_domain(Operator, Hyperparameter, Value) :-
    get_argument_by_conclusion(domain(Operator, Hyperparameter, Value)).

fetch_prototype(Prototype) :-
    get_argument_by_conclusion(pipeline_prototype(Steps, Algorithm)),
    append(Steps, [Algorithm], Prototype).

get_argument_by_conclusion(Conc) :-
    context_check(argument([_, _, [Conc], _, _])).

get_argument_by_conclusion(Conc, [A, B, [Conc], C, D]) :-
    context_check(argument([A, B, [Conc], C, D])).

get_attacked_by_conclusion(Conc, Attacker) :-
    context_check(attack(_, Attacker, [_, _, [Conc], _, _], _)).


% Attacca -> pipeline(Operators, Algorithm)
% mandatory_steps_for_algorithm([discretization], dt).

% Attacca -> pipeline_prototype(Steps, algorithm)
% incompatible_steps([normalization, discretization]).

% Attacca -> pipeline_instance(Operators, Algorithm, Hyperparameter, Comparator, Value)
% hyperparameter_exception(algorithm, dt, max_depth, eq, 1).

% Attacca -> pipeline_instance(Operators, Algorithm, Hyperparameter, Comparator, Value)
% hyperparameter_exception(algorithm, dt, max_depth, gt, 2, [normalization]).


% prendo eccezione X
%    prendo argomenti Z attaccati di tipo Y
%    argomento Z è il template


%fetch_mandatory_templates(Templates) :-
%    findall((mandatory, Steps, Algorithm), get_argument_by_conclusion(mandatory_steps_for_algorithm(Steps, Algorithm), Templates).
%    get_attacked_by_conclusion(pipeline(Operators, Algorithm), Attacker),
%    member(Operator, Operators),
%    get_argument_by_conclusion(operator(Step, Operator)),
%    \+ member(Step, Steps).

%fetch_incompatible_steps(Templates) :-
%    findall((incompatible, Steps), get_argument_by_conclusion(incompatible_steps(Steps), Templates).


%fetch_incompatible_steps(Templates) :-
%    findall((incompatible, Steps), get_argument_by_conclusion(incompatible_steps(Steps), Templates).

