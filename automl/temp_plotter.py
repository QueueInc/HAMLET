import json
import os

import matplotlib.pyplot as plt
import numpy as np

from pymoo.indicators.hv import HV


def load_resuts(working_path, hamlet_modes, abscissa_label, ordinate_label):
    results = {}
    for hamlet_mode in hamlet_modes:
        for run_idx in range(1, 5 if "ika" in hamlet_mode else 2):
            with open(
                os.path.join(
                    working_path,
                    f"{hamlet_mode}_automl_output_{run_idx}.json",
                )
            ) as f:
                results[f"{hamlet_mode}_{run_idx}"] = [
                    (
                        config[abscissa_label],
                        config[ordinate_label],
                    )
                    for config in json.load(f)["best_config"]
                ]
    return results


def plot_results(working_path, results, hamlet_modes, abscissa_label, ordinate_label):

    # Pymoo calculates HyperVolume just by minimization
    ind = HV(ref_point=np.array([1, 1]))

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    for idx_ax, hamlet_mode in enumerate(hamlet_modes):
        id_x = int(idx_ax / 2)
        id_y = int(idx_ax % 2)

        current_results = {
            key: value for key, value in results.items() if hamlet_mode == key[:-2]
        }

        hypervolumes = {}
        print()
        print()
        print(hamlet_mode)
        for key, value in current_results.items():
            print(value)
            pareto_costs_x = [result[0] for result in value]
            pareto_costs_y = [result[1] for result in value]

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
                c=f"C{key[-1]}",
                label=f"it={key[-1]}",
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
                c=f"C{key[-1]}",
            )
            hypervolumes[key[-1]] = ind(
                np.array([[1 - elem[0], 1 - elem[1]] for elem in value])
            )
            hypervolumes_text = ", ".join(
                [f"{round(elem, 2)}" for elem in hypervolumes.values()]
            )
            print([[1 - elem[0], 1 - elem[1]] for elem in value])
            print(key[-1], round(hypervolumes[key[-1]], 2))

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

    handles, labels = plt.gca().get_legend_handles_labels()
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
            working_path,
            f"summary.png",
        ),
        bbox_extra_artists=(lgd, text),
        bbox_inches="tight",
    )


def main():

    hamlet_modes = ["baseline", "pkb", "ika", "pkb_ika"]
    version = "1_0_9"
    abscissa_label = "demographic_parity"
    ordinate_label = "balanced_accuracy"
    working_path = os.path.join(
        "/",
        "home",
        "results_fairness",
        version,
    )

    results = load_resuts(
        working_path=working_path,
        hamlet_modes=hamlet_modes,
        abscissa_label=abscissa_label,
        ordinate_label=ordinate_label,
    )
    # print(results)

    plot_results(
        working_path=working_path,
        results=results,
        hamlet_modes=hamlet_modes,
        abscissa_label=abscissa_label,
        ordinate_label=ordinate_label,
    )


if __name__ == "__main__":
    main()
