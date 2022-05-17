# OpenML provides several benchmark datasets
import openml


def get_dataset(name):
    loader = {
        "blood": 1464,
        "breast": 1465,  # this is breast-tissue, not breast cancer
        "diabetes": 37,
        "ecoli": 40671,
        "iris": 61,
        "parkinsons": 1488,
        "seeds": 1499,
        "thyroid": 40682,
        "vehicle": 54,
        "wine": 187,
    }
    if name in loader:
        return load_dataset_from_openml(loader[name])
    else:
        raise Exception("There is no such a dataset in the loader")


def load_dataset_from_openml(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    dataset_features_names = [str(elem) for elem in list(dataset.features.values())]
    dataset_features_names = dataset_features_names[1:]
    return X, y, dataset_features_names
