from hyperopt import Trials, JOB_STATE_NEW

from hamlet.buffer import Buffer

from ray.tune.result import DEFAULT_METRIC
from ray.tune.suggest.hyperopt import HyperOptSearch

try:
    import hyperopt as hpo
except ImportError:
    hpo = None


class CustomHyperOptSearch(HyperOptSearch):
    def generate_trial(self, tid, space):
        idxs = {k: ([tid] if len(v) > 0 else []) for k, v in space.items()}
        vals = {k: v for k, v in space.items()}
        return {
            "state": JOB_STATE_NEW,
            "tid": tid,
            "spec": None,
            "result": {"status": "new"},
            "misc": {
                "tid": tid,
                "cmd": ("domain_attachment", "FMinIter_Domain"),
                "workdir": None,
                "idxs": idxs,
                "vals": vals,
            },
            "exp_key": None,
            "owner": None,
            "version": 0,
            "book_time": None,
            "refresh_time": None,
        }

    def generate_trials_to_calculate(self, points):
        """
        Function that generates trials to be evaluated from list of points

        :param points: List of points to be inserted in trials object in form of
            dictionary with variable names as keys and variable values as dict
            values. Example value:
            [{'x': 0.0, 'y': 0.0}, {'x': 1.0, 'y': 1.0}]

        :return: object of class base.Trials() with points which will be calculated
            before optimisation start if passed to fmin().
        """
        trials = Trials()
        new_trials = [self.generate_trial(tid, x) for tid, x in enumerate(points)]
        trials.insert_trial_docs(new_trials)
        return trials

    def _setup_hyperopt(self) -> None:

        if self._metric is None and self._mode:
            # If only a mode was passed, use anonymous metric
            self._metric = DEFAULT_METRIC

        if self._points_to_evaluate is None:
            self._hpopt_trials = hpo.Trials()
            self._points_to_evaluate = 0
        else:
            assert isinstance(self._points_to_evaluate, (list, tuple))

            new_points = list(
                reversed(
                    [Buffer().encode_conf(point) for point in self._points_to_evaluate]
                )
            )

            self._hpopt_trials = self.generate_trials_to_calculate(new_points)
            self._hpopt_trials.refresh()
            self._points_to_evaluate = len(self._points_to_evaluate)

        self.domain = hpo.Domain(lambda spc: spc, self._space)
