from .Loader import Loader


class Buffer:
    _instance = None
    _loader = None
    _configs = {}
    _results = {}
    _current_iteration = 0

    def __new__(cls, path=None):
        if cls._instance is None:
            cls._instance = super(Buffer, cls).__new__(cls)

        if path:
            cls._instance._loader = Loader(path)

        return cls._instance

    def get_space(self):
        return self._loader.get_space()

    def add_evaluation(self, config, result):
        self._configs[self._current_iteration] = config
        self._results[self._current_iteration] = result
        self._current_iteration += 1

    def get_evaluations(self):
        points_to_evaluate = (
            list(self._configs.values()) + self._loader.get_instance_constraints()
        )
        evaluated_rewards = [
            result["accuracy"] for result in self._results.values()
        ] + [float("-inf")] * len(self._loader.get_instance_constraints())
        return points_to_evaluate, evaluated_rewards

    def check_template_constraints(self, config):
        for constraint in self._loader.get_template_constraints():
            if constraint(config):
                return True
        return False

    def check_instance_constraints(self, config):
        if config in self._loader.get_instance_constraints():
            return True
        return False
