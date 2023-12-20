#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to build the ligand in this project.

Author: Andrew Tarzia

"""

import logging
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from env_set import project_path


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
        sys.exit()
    else:
        pass

    survey_path = project_path() / "CSD_survey"
    csv_file = survey_path / "NPd_survey_data_261119.csv"
    xrd_data = pd.read_csv(csv_file)
    just_names = set(xrd_data["NAME"])

    # Get bond lengths.
    distance_list = []
    for idx, row in xrd_data.iterrows():
        distance_list.append(float(row["DIST1"]))
        distance_list.append(float(row["DIST2"]))
        distance_list.append(float(row["DIST3"]))
        distance_list.append(float(row["DIST4"]))

    mean = np.mean(distance_list)
    std = np.std(distance_list)

    print("bond lengths:")
    print(f"Mean = {mean} Angstrom")
    print(f"Standard Deviation = {std} Angstrom")
    print("-------------------------------------------------------")
    X_range = (1.8, 2.2)
    width = 0.01

    # Plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    X_bins = np.arange(X_range[0], X_range[1], width)

    hist, bin_edges = np.histogram(
        a=distance_list,
        bins=X_bins,
        density=False,
    )
    ax.bar(
        bin_edges[:-1],
        hist,
        align="edge",
        alpha=1.0,
        width=width,
        color="#DAF7A6",
        edgecolor="k",
    )
    ax.axvline(x=mean, c="gray", ls="--")
    ax.text(x=1.81, y=180, s=f"{len(just_names)} structures", fontsize=16)
    ax.text(x=1.81, y=170, s=f"{len(xrd_data)} centres", fontsize=16)
    ax.text(x=1.81, y=160, s=f"{len(distance_list)} bonds", fontsize=16)
    ax.text(x=1.81, y=150, s=f"mean = {round(mean, 2)}", fontsize=16)
    ax.text(x=1.81, y=140, s=f"std. dev. = {round(std, 2)}", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"N-Pd bond distance [$\mathrm{\AA}$]", fontsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_xlim(X_range)

    fig.tight_layout()
    fig.savefig(
        survey_path / "N_Pd_bond_survey.png",
        dpi=720,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
