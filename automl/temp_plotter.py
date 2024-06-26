import traceback
import shutup

shutup.please()

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from pymoo.indicators.hv import HV

from hamlet.utils.commons import create_directory


def load_resuts(working_path, abscissa_label, ordinate_label):
    results = {}
    for hamlet_mode in [
        name
        for name in os.listdir(working_path)
        if os.path.isdir(os.path.join(working_path, name)) and name != "plots"
    ]:
        hamlet_path = os.path.join(working_path, hamlet_mode)
        results[hamlet_mode] = {}
        for dataset_id in [
            name
            for name in os.listdir(hamlet_path)
            if os.path.isdir(os.path.join(hamlet_path, name))
        ]:
            results[hamlet_mode][dataset_id] = []
            dataset_path = os.path.join(hamlet_path, dataset_id)
            output_path = os.path.join(dataset_path, "automl", "output")
            for run_idx in range(1, 5 if "ika" in hamlet_mode else 2):
                try:
                    with open(
                        os.path.join(
                            output_path,
                            f"automl_output_{run_idx}.json",
                        )
                    ) as f:
                        results[hamlet_mode][dataset_id].append(
                            [
                                (
                                    config[abscissa_label],
                                    config[ordinate_label],
                                )
                                for config in json.load(f)["best_config"]
                            ]
                        )
                except Exception as e:
                    print(
                        f"\tfunction: load_resuts\n\tdataset: {dataset_id}\n\thamlet_mode: {hamlet_mode}\n\trun_idx: {run_idx}\n\texception: {e}"
                    )
                    print()
    return results


def convert_results_structure(results):
    converted = {}
    for mode, datasets in results.items():
        for dataset, values in datasets.items():
            if dataset not in converted:
                converted[dataset] = {}
            converted[dataset][mode] = values
    return converted


def parse_elem(elem):
    return elem if elem != "'-inf'" else float("-inf")


def plot_results(working_path, results, abscissa_label, ordinate_label):

    # Pymoo calculates HyperVolume just by minimization
    ind = HV(ref_point=np.array([1, 1]))
    output_path = create_directory(os.path.join(working_path, "plots", "paretos"))
    new_results = convert_results_structure(results)

    for dataset_id, dataset_results in new_results.items():
        try:
            fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
            handles, labels = [], []
            for idx_ax, (hamlet_mode, hamlet_results) in enumerate(
                dataset_results.items()
            ):
                id_x = int(idx_ax / 2)
                id_y = int(idx_ax % 2)

                hypervolumes = {}
                hypervolumes_text = ""
                # print()
                # print()
                # print(hamlet_mode)
                for idx_it, tuple in enumerate(hamlet_results):
                    if not any([True for elem in tuple if '"-inf"' in elem]):
                        # print(tuple)
                        pareto_costs_x = [parse_elem(elem[0]) for elem in tuple]
                        pareto_costs_y = [parse_elem(elem[1]) for elem in tuple]

                        pareto_costs_x, pareto_costs_y = zip(
                            *sorted(zip(pareto_costs_x, pareto_costs_y))
                        )

                        pareto_costs_x, pareto_costs_y = (
                            list(pareto_costs_x),
                            list(pareto_costs_y),
                        )

                        ax[id_x, id_y].scatter(
                            pareto_costs_x,
                            pareto_costs_y,
                            marker="x",
                            c=f"C{idx_it}",
                            label=f"it={idx_it}",
                        )
                        ax[id_x, id_y].step(
                            [pareto_costs_x[0]]
                            + pareto_costs_x
                            + [max(pareto_costs_x)],  # We add bounds
                            [max(pareto_costs_y)]
                            + pareto_costs_y
                            + [min(pareto_costs_y)],  # We add bounds
                            where="post",
                            linestyle=":",
                            c=f"C{idx_it}",
                        )
                        hypervolumes[idx_it] = ind(
                            np.array([[1 - elem[0], 1 - elem[1]] for elem in tuple])
                        )
                        hypervolumes_text = ", ".join(
                            [f"{round(elem, 2)}" for elem in hypervolumes.values()]
                        )
                        # print([[1 - elem[0], 1 - elem[1]] for elem in tuple])
                        # print(idx_it, round(hypervolumes[idx_it], 2))

                ax[id_x, id_y].set_title(
                    f"""{hamlet_mode}
                    hv={hypervolumes_text}"""
                )
                # ref_point = np.array([0, 0])

                # ind = HV(ref_point=ref_point)
                # print(f"{hamlet_mode} it. {key[-1]} = {ind(A)}")

                if idx_ax in [2, 3]:
                    ax[id_x, id_y].set_xlabel(abscissa_label)
                if idx_ax in [0, 2]:
                    ax[id_x, id_y].set_ylabel(ordinate_label)
                # ax[id_x, id_y].legend()

                temp_handles, temp_labels = ax[id_x, id_y].get_legend_handles_labels()
                handles += temp_handles
                labels += temp_labels
            by_label = dict(zip(labels, handles))
            lgd = fig.legend(
                by_label.values(),
                by_label.keys(),
                loc="lower center",
                ncol=8,
                bbox_to_anchor=(0.5, 1.0),
            )
            text = fig.text(-0.2, 1.05, "", transform=ax[1, 1].transAxes)
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_path,
                    f"{dataset_id}.png",
                ),
                bbox_extra_artists=(lgd, text),
                bbox_inches="tight",
            )
        except Exception as e:
            print(
                f"\tfunction: plot_results\n\tdataset: {dataset_id}\n\thamlet_mode: {hamlet_mode}\n\texception: {e}"
            )
            print(traceback.print_exc())
            print(tuple)
            print()


def plot_version(version, performance_metric, fairness_metric, mining_metric):
    working_path = os.path.join(
        "/",
        "home",
        "results_fairness",
        "-".join([version, fairness_metric, mining_metric]),
        "hamlet",
    )

    print("####")
    print(f"fairness_metric: {fairness_metric}\nmining_metric: {mining_metric}")

    results = load_resuts(
        working_path=working_path,
        abscissa_label=fairness_metric,
        ordinate_label=performance_metric,
    )
    # print(results)

    plot_results(
        working_path=working_path,
        results=results,
        abscissa_label=fairness_metric,
        ordinate_label=performance_metric,
    )
    print("####")
    print()
    print()


def main():

    version = "1.0.22-fairness-dev"
    performance_metric = "balanced_accuracy"
    fairness_metric = "demographic_parity"
    mining_metric = "demographic_parity"

    for fairness_metric in ["demographic_parity", "equalized_odds"]:
        for mining_metric in [fairness_metric, ""]:
            plot_version(
                version=version,
                performance_metric=performance_metric,
                fairness_metric=fairness_metric,
                mining_metric=mining_metric,
            )


if __name__ == "__main__":
    main()
