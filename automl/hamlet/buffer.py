import signal
import sys


class TimeException(BaseException):
    pass


class Buffer:
    _instance = None

    _num_points_to_consider = None
    _metrics = None
    _template_constraints = []
    _configs = []
    _results = []
    _current_point_to_evaluate = 0
    _num_previous_evaluated_points = 0
    _initial_design_configs = 0

    def __new__(cls, metrics=None, loader=None, initial_design_configs=None):
        if cls._instance is None:
            cls._instance = super(Buffer, cls).__new__(cls)

        if metrics and loader:
            cls._instance._metrics = metrics
            cls._instance._initial_design_configs = initial_design_configs
            cls._instance._template_constraints = loader.get_template_constraints()
            (
                cls._instance._configs,
                cls._instance._results,
            ) = cls._instance._filter_previous_results(
                loader.get_points_to_evaluate(),
                loader.get_evaluated_rewards(),
                metrics,
            )

            # We put the partial configuration at the end because flaml can sample in tha subspace
            # Indeed, in check_points_to_evaluate, to detect when we sample with flaml,
            # we check the length of the th two lists _num_previous_evaluated_points and _result,
            # the former will be longer than the latter.
            cls._instance._num_previous_evaluated_points = len(
                cls._instance._configs
            ) + len(loader.get_instance_constraints())

        return cls._instance

    def _filter_previous_results(self, points_to_evaluate, evaluated_rewards, metrics):
        new_points_to_evaluate = []
        new_evaluated_rewards = []
        for i, point_to_evaluate in enumerate(points_to_evaluate):
            if self.check_template_constraints(point_to_evaluate):
                for metric in metrics:
                    evaluated_rewards[i][metric] = float("-inf")
                evaluated_rewards[i]["status"] = "previous_constraint"
                new_points_to_evaluate.append(point_to_evaluate)
                new_evaluated_rewards.append(evaluated_rewards[i])
            elif evaluated_rewards[i]["status"] != "previous_constraint":
                for metric in metrics:
                    evaluated_rewards[i][metric] = float(evaluated_rewards[i][metric])
                new_points_to_evaluate.append(point_to_evaluate)
                new_evaluated_rewards.append(evaluated_rewards[i])
        if (len(new_points_to_evaluate) != len(points_to_evaluate)) or (
            len(new_evaluated_rewards) != len(evaluated_rewards)
        ):
            raise Exception(
                "points_to_evaluate or evaluated_rewards have difrent length after Buffer processing"
            )
        return new_points_to_evaluate, new_evaluated_rewards

    def add_evaluation(self, config, result):
        if self._current_point_to_evaluate < (self._initial_design_configs + 1):
            self._configs.insert(0, config)
            self._results.insert(0, result)
        else:
            self._configs.append(config)
            self._results.append(result)
        self.printflush(len(self._configs))

    def get_evaluations(self):
        return self._configs.copy(), self._results.copy()

    def check_template_constraints(self, config):
        for constraint in self._template_constraints:
            if constraint(config):
                return True
        return False

    def check_points_to_evaluate(self):
        if self._current_point_to_evaluate < self._initial_design_configs:
            self._current_point_to_evaluate += 1
            return False, 0
        elif self._current_point_to_evaluate < (
            self._initial_design_configs + self._num_previous_evaluated_points
        ):
            # print(
            #     f"{self._current_point_to_evaluate} out of {self._num_previous_evaluated_points}"
            # )
            self._current_point_to_evaluate += 1
            # Here, we return the -inf result if flaml is sampling (see comment above in the constructor)
            if self._current_point_to_evaluate < (len(self._results) + 1):
                return True, self._results[self._current_point_to_evaluate - 1]
            return True, {
                **{metric: float("-inf") for metric in self._metrics},
                **{"status": "previous_constraint"},
            }
        return False, 0

    def printflush(self, message):
        print(message)
        sys.stdout.flush()

    def attach_handler(self):
        def handler(signum, frame):
            raise TimeException("Timeout Exception")

        signal.signal(signal.SIGALRM, handler)

    def attach_timer(self, time_in_s):
        signal.alarm(time_in_s)

    def detach_timer(self):
        signal.alarm(0)
