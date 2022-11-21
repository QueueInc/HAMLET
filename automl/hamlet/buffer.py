class Buffer:
    _instance = None

    _num_points_to_consider = None
    _metric = None
    _template_constraints = []
    _configs = []
    _results = []
    _current_point_to_evaluate = 0
    _num_previous_evaluated_points = 0

    def __new__(cls, metric=None, loader=None):
        if cls._instance is None:
            cls._instance = super(Buffer, cls).__new__(cls)

        if metric and loader:
            cls._instance._metric = metric
            cls._instance._template_constraints = loader.get_template_constraints()
            (
                cls._instance._configs,
                cls._instance._results,
            ) = cls._instance._filter_previous_results(
                loader.get_points_to_evaluate(),
                loader.get_evaluated_rewards(),
                metric,
            )

            cls._instance._num_previous_evaluated_points = len(
                cls._instance._configs
            ) + len(loader.get_instance_constraints())

        return cls._instance

    def _filter_previous_results(self, points_to_evaluate, evaluated_rewards, metric):
        new_points_to_evaluate, new_evaluated_rewards = [], []
        for i, point_to_evaluate in enumerate(points_to_evaluate):
            new_points_to_evaluate.append(point_to_evaluate)
            if self.check_template_constraints(point_to_evaluate):
                new_evaluated_rewards.append(
                    {metric: float("-inf"), "status": "previous_constraint"}
                )
            else:
                if evaluated_rewards[i]["status"] != "previous_constraint":
                    new_evaluated_rewards.append(
                        {
                            metric: float(evaluated_rewards[i][metric]),
                            "status": evaluated_rewards[i]["status"],
                        }
                    )
        return new_points_to_evaluate, new_evaluated_rewards

    def add_evaluation(self, config, result):
        self._configs.append(config)
        self._results.append(result)
        print(len(self._configs))

    def get_evaluations(self):
        return self._configs.copy(), self._results.copy()

    def check_template_constraints(self, config):
        for constraint in self._template_constraints:
            if constraint(config):
                return True
        return False

    def check_points_to_evaluate(self):
        if self._current_point_to_evaluate < self._num_previous_evaluated_points:
            print(
                f"{self._current_point_to_evaluate} out of {self._num_previous_evaluated_points}"
            )
            self._current_point_to_evaluate += 1
            if self._current_point_to_evaluate < len(self._results):
                return True, self._results[self._current_point_to_evaluate]
            return True, {self._metric: float("-inf"), "status": "previous_constraint"}
        return False, 0
