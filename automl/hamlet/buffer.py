from .loader import Loader


class Buffer:
    _instance = None
    _loader = None

    _num_points_to_consider = None
    _configs = []
    _results = []
    _current_point_to_evaluate = 0
    _max_points_to_evaluates = 0

    def __new__(cls, metric=None, input_path=None):
        if cls._instance is None:
            cls._instance = super(Buffer, cls).__new__(cls)

        if metric and input_path:
            cls._instance._loader = Loader(input_path)

            (
                points_to_evaluate,
                evaluated_rewards,
            ) = cls._instance._filter_previous_results(
                cls._instance._loader.get_points_to_evaluate(),
                cls._instance._loader.get_evaluated_rewards(),
                metric,
            )

            cls._instance._num_points_to_consider = len(points_to_evaluate)
            cls._instance._configs = (
                cls._instance._loader.get_instance_constraints() + points_to_evaluate
            )
            cls._instance._max_points_to_evaluates = len(cls._instance._configs)
            cls._instance._results = [
                {metric: float("-inf"), "status": "previous_constraint"}
            ] * len(cls._instance._loader.get_instance_constraints()) + [
                {
                    metric: float(reward[metric]),
                    "status": reward["status"],
                }
                for reward in evaluated_rewards
            ]
        return cls._instance

    def _filter_previous_results(self, points_to_evaluate, evaluated_rewards, metric):
        new_points_to_evaluate, new_evaluated_rewards = [], []
        for i in range(len(points_to_evaluate)):
            if self.check_template_constraints(points_to_evaluate[i]):
                new_points_to_evaluate.append(points_to_evaluate[i])
                new_evaluated_rewards.append(
                    {metric: float("-inf"), "status": "previous_constraint"}
                )
            else:
                if evaluated_rewards[i]["status"] != "previous_constraint":
                    new_points_to_evaluate.append(points_to_evaluate[i])
                    new_evaluated_rewards.append(evaluated_rewards[i])
        return new_points_to_evaluate, new_evaluated_rewards

    def get_num_points_to_consider(self):
        return self._num_points_to_consider

    def get_space(self):
        return self._loader.get_space()

    def add_evaluation(self, config, result):
        self._configs.append(config)
        self._results.append(result)
        self._num_points_to_consider += 1

    def get_evaluations(self):
        return self._configs.copy(), self._results.copy()

    def check_template_constraints(self, config):
        for constraint in self._loader.get_template_constraints():
            if constraint(config):
                return True
        return False

    def check_points_to_evaluate(self, config):
        if self._current_point_to_evaluate < self._max_points_to_evaluates:
            self._configs[self._current_point_to_evaluate] = config
            to_return = self._results[self._current_point_to_evaluate]
            self._current_point_to_evaluate += 1
            return True, to_return
        return False, 0
