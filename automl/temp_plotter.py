import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_resuts(working_path, hamlet_modes):
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
                        config["demographic_parity"],
                        config["balanced_accuracy"],
                    )
                    for config in json.load(f)["best_config"]
                ]
    return results


def plot_results(working_path, results, hamlet_modes):

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    for idx_ax, hamlet_mode in enumerate(hamlet_modes):
        id_x = int(idx_ax / 2)
        id_y = int(idx_ax % 2)

        print(hamlet_mode)
        current_results = {
            key: value for key, value in results.items() if hamlet_mode == key[:-2]
        }

        print(current_results)

        for key, value in current_results.items():
            pareto_costs_x = [result[0] for result in value]
            pareto_costs_y = [result[1] for result in value]

            ax[id_x, id_y].scatter(
                pareto_costs_x,
                pareto_costs_y,
                marker="x",
                c=f"C{key[-1]}",
                label=f"{key}",
            )
            # ax[id_x, id_y].step(
            #     [pareto_costs_x[0]] + pareto_costs_x + [1],  # We add bounds
            #     [1] + pareto_costs_y + [0],  # We add bounds
            #     where="post",
            #     linestyle=":",
            #     c="r",
            # )

        ax[id_x, id_y].set_title(f"{hamlet_mode}")
        ax[id_x, id_y].legend()
    fig.savefig(
        os.path.join(
            working_path,
            f"trial.png",
        )
    )


def main():

    hamlet_modes = ["baseline", "pkb", "ika", "pkb_ika"]
    working_path = os.path.join(
        "/",
        "home",
        "results_fairness",
        "cluster",
    )

    results = load_resuts(working_path=working_path, hamlet_modes=hamlet_modes)
    # print(results)

    plot_results(working_path=working_path, results=results, hamlet_modes=hamlet_modes)


if __name__ == "__main__":
    main()
