from .loader import Loader


class Buffer:
    _instance = None
    _loader = None
    _configs = []
    _results = []

    def __new__(cls, metric=None, input_path=None):
        if cls._instance is None:
            cls._instance = super(Buffer, cls).__new__(cls)

        if metric and input_path:
            cls._instance._loader = Loader(input_path)
            cls._instance._configs = (
                cls._instance._loader.get_points_to_evaluate()
                + cls._instance._loader.get_instance_constraints()
            )
            cls._instance._results = [
                {metric: float(result), "status": "success"}
                for result in cls._instance._loader.get_evaluated_rewards()
            ] + [{metric: float("-inf"), "status": "fail"}] * len(
                cls._instance._loader.get_instance_constraints()
            )
        return cls._instance

    def get_space(self):
        return self._loader.get_space()

    def add_evaluation(self, config, result):
        self._configs.append(config)
        self._results.append(result)

    def get_evaluations(self):
        return self._configs.copy(), self._results.copy()

    def check_template_constraints(self, config):
        for constraint in self._loader.get_template_constraints():
            if constraint(config):
                return True
        return False

    def check_points_to_evaluate(self, config):
        if config in self._configs:
            return True, self._results[self._configs.index(config)]
        return False, 0
