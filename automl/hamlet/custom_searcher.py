from hyperopt import Trials, JOB_STATE_NEW

from ray.tune.result import DEFAULT_METRIC
from ray.tune.suggest.hyperopt import HyperOptSearch

try:
    import hyperopt as hpo
except ImportError:
    hpo = None

from utils.json_to_csv import flattenjson


class CustomHyperOptSearch(HyperOptSearch):
    def __init__(self, custom_space=None, *args, **kwargs):
        self._custom_space = custom_space
        super().__init__(*args, **kwargs)

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
                    [self._encode_conf(point) for point in self._points_to_evaluate]
                )
            )

            self._hpopt_trials = self._generate_trials_to_calculate(new_points)
            self._hpopt_trials.refresh()
            self._points_to_evaluate = len(self._points_to_evaluate)

        self.domain = hpo.Domain(lambda spc: spc, self._space)

    def _generate_trial(self, tid, space):
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

    def _generate_trials_to_calculate(self, points):
        trials = Trials()
        new_trials = [self._generate_trial(tid, x) for tid, x in enumerate(points)]
        trials.insert_trial_docs(new_trials)
        return trials

    def _encode_conf(self, conf):
        result = {}
        flatten_conf = flattenjson(conf, "/")
        mapped_space = {k: v for k, v in self._get_keys(self._custom_space, "")}
        for key in mapped_space.keys():
            if key in flatten_conf.keys():
                if mapped_space[key][0] != "choice":
                    result[key] = [flatten_conf[key]]
                else:
                    if flatten_conf[key] in mapped_space[key][1]:
                        result[key] = [mapped_space[key][1].index(flatten_conf[key])]
                    else:
                        result[key] = []
            else:
                if "/" in key:
                    result[key] = []
                else:
                    if flatten_conf[key + "/type"] in mapped_space[key][1]:
                        result[key] = [
                            mapped_space[key][1].index(flatten_conf[key + "/type"]) + 1
                        ]
                    else:
                        result[key] = [0]

        return result

    def _get_keys(self, input_space, path):
        def convert_value(value):
            if any((type(x) is dict) for x in value):
                return [x["type"] for x in value if type(x["type"]) is not dict]
            return value

        values = []

        if type(input_space) is not dict:
            return []
        for key, value in input_space.items():
            if key == "choice":
                return [x for elem in value for x in self._get_keys(elem, path)] + [
                    (path, ("choice", convert_value(value)))
                ]
            if key == "randint":
                return [(path, ("randint", value))]
            if type(value) is dict:
                values = values + self._get_keys(
                    value, (path + "/" + key) if path else key
                )
            else:
                pass
        return values
