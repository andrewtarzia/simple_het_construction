#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for plotting functions.

Author: Andrew Tarzia

"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from matplotlib.lines import Line2D
import matplotlib

import logging
import numpy as np
import os

from env_set import figu_path
from utilities import name_parser, name_conversion


def c_and_m_properties():
    return {
        "m2": ("#212738", "o"),
        "m3": ("#212738", "o"),
        "m4": ("#212738", "o"),
        "m6": ("#212738", "o"),
        "m12": ("#212738", "o"),
        "m24": ("#212738", "o"),
        "m30": ("#212738", "o"),
        "cis": ("#F97068", "o"),
        "trans": ("#57C4E5", "o"),
    }


def axes_labels(prop):
    def no_conv(y):
        return y

    def avg_strain(y):
        return np.mean(list(y.values())) * 2625.5

    return {
        "min_order_param": ("min op.", (0, 1), no_conv),
        "pore_diameter_opt": (
            r"pywindow pore volume [$\mathrm{\AA}^{3}$]",
            (0, 20),
            no_conv,
            None,
        ),
        "xtb_dmsoenergy": (
            "xtb/DMSO energy [kJmol-1]",
            (0, 20),
            no_conv,
            None,
        ),
        "xtb_lig_strain_au": (
            "avg. xtb/DMSO strain energy [kJ mol-1]",
            (0, 20),
            avg_strain,
            (-20, 20),
        ),
        "xtb_energy_au": (
            "xtb energy [kJ mol-1]",
            None,
            no_conv,
        ),
        "xtb_solv_opt_dmsoenergy_au": (
            "xtb/DMSO energy [kJ mol-1]",
            (-400, -200),
            no_conv,
            None,
        ),
        "pbe0_def2svp_sp_gas_kjmol": (
            "PBE0/def2-svp/GD3BJ/gas SP energy [kJ mol-1]",
            (-400, -200),
            no_conv,
            None,
        ),
        "pbe0_def2svp_sp_dmso_kjmol": (
            "PBE0/def2-svp/GD3BJ/DMSO SP energy [kJ mol-1]",
            (-400, -200),
            no_conv,
            None,
        ),
        "pbe0_def2svp_opt_gas_kjmol": (
            "PBE0/def2-svp/GD3BJ/gas OPT energy [kJ mol-1]",
            (-400, -200),
            no_conv,
            None,
        ),
        "pbe0_def2svp_opt_dmso_kjmol": (
            "PBE0/def2-svp/GD3BJ/DMSO OPT energy [kJ mol-1]",
            (-400, -200),
            no_conv,
            None,
        ),
        "NcentroidN_angle": (
            "N-centroid-N angle [deg]",
            (90, 180),
            no_conv,
        ),
        "NN_distance": (
            "N-N distance [A]",
            (0, 30),
            no_conv,
        ),
        "NN_BCN_angles": (
            "theta [deg]",
            (0, 180),
            no_conv,
        ),
        "bite_angle": (
            "bite angle [deg]",
            (-180, 180),
            no_conv,
        ),
        "NCCN_dihedral": (
            "abs. NCCN dihedral [deg]",
            (0, 180),
            no_conv,
        ),
        "avg_heli": (
            "avg. helicity [deg]",
            (0, 180),
            no_conv,
        ),
        "min_heli": (
            "min. helicity [deg]",
            (0, 180),
            no_conv,
        ),
        "max_heli": (
            "max. helicity [deg]",
            (0, 220),
            no_conv,
        ),
        "pore_angle": (
            "M-centroid-M [deg]",
            (0, 185),
            no_conv,
        ),
        "mm_distance": (
            "MM distance [A]",
            (0, 30),
            no_conv,
        ),
    }[prop]


def plot_geom_scores_vs_threshold(
    results_dict,
    dihedral_cutoff,
    outname,
    experimental_ligand_outcomes,
):
    logging.info("plotting: plot_geom_scores_vs_threshold")

    fig, axs = plt.subplots(
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=(10, 5),
    )
    # fig, ax = plt.subplots(figsize=(8, 5))

    xs = np.linspace(0, 1, 30)
    colour_map = {
        "forms cis-cage": "#086788",
        "does not form cis-cage": "#F9A03F",
        "this work": "k",
    }

    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            ename = pair_name.split(",")
            if (ename[0], ename[1]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[0], ename[1])]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[1], ename[0])]
            else:
                continue
            if edata == "yes":
                colour = colour_map["forms cis-cage"]
                ax = axs[1]
            elif edata == "no":
                colour = colour_map["does not form cis-cage"]
                ax = axs[1]
        else:
            colour = colour_map["this work"]
            ax = axs[0]

        ys = []
        # ys_l = []
        # ys_a = []
        for threshold in xs:
            all_scores = 0
            less_scores = 0
            # less_l_scores = 0
            # less_a_scores = 0
            for cid_pair in rdict:
                all_scores += 1

                if (
                    abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                    or abs(rdict[cid_pair]["small_dihedral"]) > dihedral_cutoff
                ):
                    continue

                geom_score = rdict[cid_pair]["geom_score"]
                if geom_score < threshold:
                    less_scores += 1

                # angle_score = rdict[cid_pair]["angle_deviation"]
                # if abs(angle_score - 1) < threshold:
                #     less_a_scores += 1

                # length_score = rdict[cid_pair]["length_deviation"]
                # if abs(length_score - 1) < threshold:
                #     less_l_scores += 1

            if all_scores == 0:
                ys.append(0)
            else:
                ys.append((less_scores / all_scores) * 100)

            # if all_scores == 0:
            #     ys_l.append(0)
            # else:
            #     ys_l.append((less_l_scores / all_scores) * 100)

            # if all_scores == 0:
            #     ys_a.append(0)
            # else:
            #     ys_a.append((less_a_scores / all_scores) * 100)

        ax.plot(
            xs,
            ys,
            lw=2,
            c=colour,
            # label=pair_name,
        )
        logging.info(f"{pair_name} has max gpercent {max(ys)}")
        # axs[1].plot(
        #     xs,
        #     ys_l,
        #     lw=2,
        #     c=colour,
        # )
        # axs[2].plot(
        #     xs,
        #     ys_a,
        #     lw=2,
        #     c=colour,
        # )

    # axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel(r"$g_{\mathrm{threshold}}$", fontsize=16)
    # ax.set_ylabel(r"% $g$ < $g_{\mathrm{threshold}}$", fontsize=16)
    axs[0].set_ylabel(r"$g_{\mathrm{percent}}$", fontsize=16)
    axs[0].axvline(x=0.45, c="gray", linestyle="--", lw=1)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel(r"$g_{\mathrm{threshold}}$", fontsize=16)
    axs[1].set_ylabel(r"$g_{\mathrm{percent}}$", fontsize=16)
    axs[1].axvline(x=0.45, c="gray", linestyle="--", lw=1)

    # # axs[1].set_ylim(0, 101)
    # axs[1].tick_params(axis="both", which="major", labelsize=16)
    # axs[1].set_xlabel("length score threshold", fontsize=16)
    # axs[1].set_ylabel(r"% good", fontsize=16)

    # # axs[2].set_ylim(0, 101)
    # axs[2].tick_params(axis="both", which="major", labelsize=16)
    # axs[2].set_xlabel("angle score threshold", fontsize=16)
    # axs[2].set_ylabel(r"% good", fontsize=16)
    legend_elements = []
    for i in colour_map:
        legend_elements.append(
            Patch(
                facecolor=colour_map[i],
                label=i,
                alpha=1.0,
                edgecolor="k",
            ),
        )

    axs[0].legend(handles=legend_elements, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_geom_scores_mean(
    results_dict,
    outname,
    dihedral_cutoff,
    strain_cutoff,
    experimental_ligand_outcomes,
):
    logging.info("plotting: plot_all_geom_scores_mean")

    colour_map = {
        "forms cis-cage": "#086788",
        "does not form cis-cage": "white",
        "this work": "#F9A03F",
    }

    fig, axs = plt.subplots(
        nrows=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )

    categories_gs = {}
    categories_lengths = {}
    categories_angles = {}
    categories_gs_stds = {}
    categories_lengths_stds = {}
    categories_angles_stds = {}
    colour_list = []

    crosses = {}
    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            ename = pair_name.split(",")
            if (ename[0], ename[1]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[0], ename[1])]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[1], ename[0])]
            else:
                continue
            if edata == "yes":
                colour_list.append(colour_map["forms cis-cage"])
            elif edata == "no":
                colour_list.append(colour_map["does not form cis-cage"])
        else:
            colour_list.append(colour_map["this work"])

        geom_scores = []
        angle_scores = []
        length_scores = []
        for cid_pair in rdict:
            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            angle_score = abs(rdict[cid_pair]["angle_deviation"] - 1)
            length_score = abs(rdict[cid_pair]["length_deviation"] - 1)
            geom_scores.append(geom_score)
            length_scores.append(length_score)
            angle_scores.append(angle_score)

        categories_gs[pair_name] = np.mean(geom_scores)
        categories_lengths[pair_name] = np.mean(length_scores)
        categories_angles[pair_name] = np.mean(angle_scores)
        categories_gs_stds[pair_name] = np.std(geom_scores)
        categories_lengths_stds[pair_name] = np.std(length_scores)
        categories_angles_stds[pair_name] = np.std(angle_scores)

    axs[0].bar(
        categories_gs.keys(),
        categories_gs.values(),
        yerr=categories_gs_stds.values(),
        color=colour_list,
        edgecolor="k",
        lw=2,
    )

    axs[1].bar(
        categories_gs.keys(),
        categories_gs.values(),
        color="gray",
        edgecolor="k",
        lw=2,
        alpha=0.5,
    )
    axs[1].bar(
        categories_lengths.keys(),
        categories_lengths.values(),
        yerr=categories_lengths_stds.values(),
        color=colour_list,
        edgecolor="k",
        lw=2,
    )

    axs[2].bar(
        categories_gs.keys(),
        categories_gs.values(),
        color="gray",
        edgecolor="k",
        lw=2,
        alpha=0.5,
    )
    axs[2].bar(
        categories_angles.keys(),
        categories_angles.values(),
        yerr=categories_angles_stds.values(),
        color=colour_list,
        edgecolor="k",
        lw=2,
    )

    for x, tstr in enumerate(categories_gs):
        if tstr in crosses:
            for ax in axs:
                ax.scatter(
                    x,
                    0.2,
                    c=crosses[tstr][0],
                    marker=crosses[tstr][1],
                    edgecolors="k",
                    s=200,
                )
    #     ax.text(
    #         x=x - 0.3,
    #         y=categories_gs[tstr] + 0.1,
    #         s=round(categories_strain[tstr], 0),
    #         c="k",
    #         fontsize=16,
    #     )

    # ax.axvline(x=4.5, linestyle="--", c="gray", lw=2)

    # ax.set_xlabel("topology", fontsize=16)
    # ax.set_ylim(0, 100)
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_ylabel(r"$g_{\mathrm{avg}}$", fontsize=16)
    # axs[0].set_xticks(range(len(categories_gs)))
    # axs[0].set_xticklabels(categories_gs.keys(), rotation=45)
    # axs[0].set_ylim(0, 2)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    # axs[1].set_ylabel("length deviation ", fontsize=16)
    axs[1].set_ylabel("average($|l-1|$)", fontsize=16)
    # axs[1].set_xticks(range(len(categories_gs)))
    # axs[1].set_xticklabels(categories_gs.keys(), rotation=45)
    # axs[1].set_ylim(0, 2)

    axs[2].tick_params(axis="both", which="major", labelsize=16)
    # axs[2].set_ylabel("angle deviation", fontsize=16)
    axs[2].set_ylabel("average($|a-1|$)", fontsize=16)
    axs[2].set_xticks(range(len(categories_gs)))
    axs[2].set_xticklabels(categories_gs.keys(), rotation=45)
    # axs[2].set_ylim(0, 2)

    legend_elements = []
    for i in colour_map:
        legend_elements.append(
            Patch(
                facecolor=colour_map[i],
                label=i,
                alpha=1.0,
                edgecolor="k",
            ),
        )

    axs[0].legend(handles=legend_elements, ncol=5, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_geom_scores_categ(
    results_dict,
    outname,
    dihedral_cutoff,
    strain_cutoff,
    experimental_ligand_outcomes,
):
    logging.info("plotting: plot_all_geom_scores_categorical")

    colour_map = {
        "previous work": "#086788",
        # "experimental": "white",
        "this work": "#F9A03F",
    }

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    flat_axs = axs.flatten()

    categories_gcs = []
    categories_gs = []
    colour_list = []
    x_list = []

    crosses = {}
    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            ename = pair_name.split(",")
            if (ename[0], ename[1]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[0], ename[1])]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[1], ename[0])]
            else:
                continue
            if edata == "yes":
                x_list.append(1)
            elif edata == "no":
                x_list.append(2)
                # colour_list.append(colour_map["does not form cis-cage"])
            colour_list.append(colour_map["previous work"])
        else:
            colour_list.append(colour_map["this work"])
            if pair_name in ("l1,lb", "l1,lc"):
                x_list.append(1)
            else:
                x_list.append(2)

        if len(rdict) == 0:
            crosses[pair_name] = (colour_list[-1], "X")
            categories_gs.append(-1)
            categories_gcs.append(-1)
            continue

        all_scores = 0
        min_geom_score = 1e24
        geom_scores = []
        for cid_pair in rdict:
            all_scores += 1

            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            geom_scores.append(geom_score)
            if geom_score < min_geom_score:
                min_geom_score = geom_score

        if all_scores == 0 or min_geom_score == 1e24:
            categories_gs.append(0)
            categories_gcs.append(1)
        else:
            categories_gs.append(min_geom_score)
            categories_gcs.append(np.mean(geom_scores))

        # flat_axs[0].text(
        #     x_list[-1] - 0.5,
        #     categories_gs[-1],
        #     s=pair_name,
        # )
        # flat_axs[1].text(
        #     x_list[-1] - 0.5,
        #     categories_gcs[-1],
        #     s=pair_name,
        # )

    flat_axs[0].scatter(
        [i + np.random.uniform(-1, 1, 1) * 0.3 for i in x_list],
        categories_gs,
        color=colour_list,
        edgecolor="k",
        s=120,
    )
    flat_axs[0].axvline(x=1.5, c="gray", linestyle="--")
    flat_axs[0].axvline(x=2.5, c="gray", linestyle="--")
    flat_axs[0].tick_params(axis="both", which="major", labelsize=16)
    flat_axs[0].set_ylabel(r"$g_{\mathrm{min}}$", fontsize=16)
    flat_axs[0].set_xlim(0.5, 2.5)
    flat_axs[0].set_xticks((1, 2))
    # flat_axs[0].set_xticklabels(("this work", "forms", "does not form"))
    flat_axs[0].set_xticklabels(("forms", "does not form"))
    flat_axs[0].set_ylim(0, 1.5)

    flat_axs[1].scatter(
        [i + np.random.uniform(-1, 1, 1) * 0.3 for i in x_list],
        categories_gcs,
        color=colour_list,
        edgecolor="k",
        s=120,
    )
    flat_axs[1].axvline(x=1.5, c="gray", linestyle="--")
    flat_axs[1].axvline(x=2.5, c="gray", linestyle="--")
    flat_axs[1].tick_params(axis="both", which="major", labelsize=16)
    flat_axs[1].set_ylabel(r"$g_{\mathrm{avg}}$", fontsize=16)
    flat_axs[1].set_xlim(0.5, 2.5)
    flat_axs[1].set_xticks((1, 2))
    flat_axs[1].set_xticklabels(("forms", "does not form"))
    flat_axs[1].set_ylim(0, 1.5)

    legend_elements = []
    for i in colour_map:
        legend_elements.append(
            Patch(
                facecolor=colour_map[i],
                label=i,
                alpha=1.0,
                edgecolor="k",
            ),
        )
    flat_axs[0].legend(handles=legend_elements, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_geom_scores(
    results_dict,
    outname,
    dihedral_cutoff,
    strain_cutoff,
    experimental_ligand_outcomes,
):
    logging.info("plotting: plot_all_geom_scores")

    colour_map = {
        "forms cis-cage": "#086788",
        "does not form cis-cage": "white",
        "this work": "#F9A03F",
    }

    fig, axs = plt.subplots(
        nrows=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )

    categories_gs = {}
    categories_lengths = {}
    categories_angles = {}
    colour_list = []

    crosses = {}
    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            ename = pair_name.split(",")
            if (ename[0], ename[1]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[0], ename[1])]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[1], ename[0])]
            else:
                continue
            if edata == "yes":
                colour_list.append(colour_map["forms cis-cage"])
            elif edata == "no":
                colour_list.append(colour_map["does not form cis-cage"])
        else:
            colour_list.append(colour_map["this work"])

        if len(rdict) == 0:
            crosses[pair_name] = (colour_list[-1], "X")
            categories_gs[pair_name] = 0
            categories_lengths[pair_name] = 0
            categories_angles[pair_name] = 0
            continue

        min_geom_score = 1e24
        min_length_score = 1e24
        min_angle_score = 1e24
        for cid_pair in rdict:
            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            angle_score = abs(rdict[cid_pair]["angle_deviation"] - 1)
            length_score = abs(rdict[cid_pair]["length_deviation"] - 1)
            if geom_score < min_geom_score:
                min_geom_score = geom_score
                min_angle_score = angle_score
                min_length_score = length_score

        if min_geom_score == 1e24:
            crosses[pair_name] = (colour_list[-1], "o")
            categories_gs[pair_name] = 0
            categories_lengths[pair_name] = 0
            categories_angles[pair_name] = 0
            continue

        categories_gs[pair_name] = min_geom_score
        categories_lengths[pair_name] = min_length_score
        categories_angles[pair_name] = min_angle_score

    axs[0].bar(
        categories_gs.keys(),
        categories_gs.values(),
        color=colour_list,
        edgecolor="k",
        lw=2,
    )

    axs[1].bar(
        categories_gs.keys(),
        categories_gs.values(),
        color="gray",
        edgecolor="k",
        lw=2,
        alpha=0.5,
    )
    axs[1].bar(
        categories_lengths.keys(),
        categories_lengths.values(),
        color=colour_list,
        edgecolor="k",
        lw=2,
    )

    axs[2].bar(
        categories_gs.keys(),
        categories_gs.values(),
        color="gray",
        edgecolor="k",
        lw=2,
        alpha=0.5,
    )
    axs[2].bar(
        categories_angles.keys(),
        categories_angles.values(),
        color=colour_list,
        edgecolor="k",
        lw=2,
    )

    for x, tstr in enumerate(categories_gs):
        if tstr in crosses:
            for ax in axs:
                ax.scatter(
                    x,
                    0.2,
                    c=crosses[tstr][0],
                    marker=crosses[tstr][1],
                    edgecolors="k",
                    s=200,
                )
    #     ax.text(
    #         x=x - 0.3,
    #         y=categories_gs[tstr] + 0.1,
    #         s=round(categories_strain[tstr], 0),
    #         c="k",
    #         fontsize=16,
    #     )

    # ax.axvline(x=4.5, linestyle="--", c="gray", lw=2)

    # ax.set_xlabel("topology", fontsize=16)
    # ax.set_ylim(0, 100)
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_ylabel(r"$g_{\mathrm{min}}$", fontsize=16)
    # axs[0].set_xticks(range(len(categories_gs)))
    # axs[0].set_xticklabels(categories_gs.keys(), rotation=45)
    # axs[0].set_ylim(0, 2)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    # axs[1].set_ylabel("length deviation ", fontsize=16)
    axs[1].set_ylabel("$|l-1|$", fontsize=16)
    # axs[1].set_xticks(range(len(categories_gs)))
    # axs[1].set_xticklabels(categories_gs.keys(), rotation=45)
    # axs[1].set_ylim(0, 2)

    axs[2].tick_params(axis="both", which="major", labelsize=16)
    # axs[2].set_ylabel("angle deviation", fontsize=16)
    axs[2].set_ylabel("$|a-1|$", fontsize=16)
    axs[2].set_xticks(range(len(categories_gs)))
    axs[2].set_xticklabels(categories_gs.keys(), rotation=45)
    # axs[2].set_ylim(0, 2)

    legend_elements = []
    for i in colour_map:
        legend_elements.append(
            Patch(
                facecolor=colour_map[i],
                label=i,
                alpha=1.0,
                edgecolor="k",
            ),
        )

    # legend_elements.append(
    #     Line2D(
    #         [0],
    #         [0],
    #         marker="o",
    #         color="w",
    #         label="failed",
    #         markerfacecolor="k",
    #         markersize=15,
    #         markeredgecolor="k",
    #     ),
    # )

    # legend_elements.append(
    #     Line2D(
    #         [0],
    #         [0],
    #         marker="X",
    #         color="w",
    #         label="failed angles",
    #         markerfacecolor="k",
    #         markersize=15,
    #         markeredgecolor="k",
    #     ),
    # )

    axs[0].legend(handles=legend_elements, ncol=5, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def gs_table(results_dict, dihedral_cutoff, strain_cutoff):
    logging.info("plotting: making gs table")

    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            continue

        min_geom_score = 1e24
        good_geoms = 0
        total_tested = 0
        geom_scores = []
        for cid_pair in rdict:
            total_tested += 1

            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            geom_scores.append(geom_score)
            if geom_score < min_geom_score:
                min_geom_score = geom_score

            if geom_score < 0.45:
                good_geoms += 1

        logging.info(
            (
                f"{pair_name}: {round(min_geom_score, 2)}, "
                f"{round((good_geoms/total_tested)*100, 0)} "
                f"{round(np.mean(geom_scores), 2)} "
                f"({round(np.std(geom_scores), 2)}) "
            )
        )


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    cbar_kw=None,
    cbarlabel="",
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=-30,
        ha="right",
        rotation_mode="anchor",
    )

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def gs_table_plot(results_dict, dihedral_cutoff, strain_cutoff, prefix):
    logging.info("plotting: gs table")

    fig, axs = plt.subplots(ncols=2, figsize=(8, 3))
    flat_axs = axs.flatten()

    matrix1 = np.zeros(shape=(3, 4))
    matrix2 = np.zeros(shape=(3, 4))
    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            continue
        else:
            # if pair_name in ("l1,lb", "l1,lc"):
            #     colour = colour_map["forms cis-cage"]
            #     x = 1
            # else:
            #     colour = colour_map["does not"]
            #     x = 2
            if "l1" in pair_name:
                # marker = marker_map["l1"]
                m_i = 0
            elif "l2" in pair_name:
                # marker = marker_map["l2"]
                m_i = 1
            elif "l3" in pair_name:
                # marker = marker_map["l3"]
                m_i = 2
            if "la" in pair_name:
                m_j = 0
            elif "lb" in pair_name:
                m_j = 1
            elif "lc" in pair_name:
                m_j = 2
            elif "ld" in pair_name:
                m_j = 3

        min_geom_score = 1e24
        total_tested = 0
        geom_scores = []
        for cid_pair in rdict:
            total_tested += 1

            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            geom_scores.append(geom_score)
            if geom_score < min_geom_score:
                min_geom_score = geom_score

        matrix1[m_i][m_j] = min_geom_score
        matrix2[m_i][m_j] = np.mean(geom_scores)

    rows = [
        name_conversion()["l1"],
        name_conversion()["l2"],
        name_conversion()["l3"],
    ]
    cols = [
        name_conversion()["la"],
        name_conversion()["lb"],
        name_conversion()["lc"],
        name_conversion()["ld"],
    ]
    im, cbar = heatmap(
        matrix1,
        rows,
        cols,
        ax=flat_axs[0],
        cmap="magma_r",
        cbarlabel=r"$g_{\mathrm{min}}$",
        vmin=0,
        vmax=1.5,
    )
    _ = annotate_heatmap(im, valfmt="{x:.2f}")
    im, cbar = heatmap(
        matrix2,
        rows,
        cols,
        ax=flat_axs[1],
        cmap="magma_r",
        cbarlabel=r"$g_{\mathrm{avg}}$",
        vmin=0,
        vmax=1.5,
    )
    _ = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{prefix}_g_tables.png"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def plot_conformer_props(
    structure_results,
    outname,
    dihedral_cutoff,
    strain_cutoff,
    low_energy_values,
):
    logging.info("plotting: plot_conformer_props")

    fig, ax = plt.subplots(figsize=(8, 5))

    for ligand in structure_results:
        if "e" in ligand:
            continue
        if "ll" in ligand:
            continue
        if "ls" in ligand:
            continue
        original_number = 500
        sres = structure_results[ligand]
        after_rmsd = len(sres)

        low_energy = low_energy_values[ligand][1]

        within_strain = {}
        for cid in sres:
            strain = sres[cid]["UFFEnergy;kj/mol"] - low_energy
            if strain <= strain_cutoff:
                within_strain[cid] = sres[cid]
        after_strain = len(within_strain)

        within_torsion = {}
        for cid in within_strain:
            dihedral = within_strain[cid]["NCCN_dihedral"]
            if abs(dihedral) <= dihedral_cutoff:
                within_torsion[cid] = within_strain[cid]

        after_torsion = len(within_torsion)
        print(ligand, len(sres))
        print(original_number, after_rmsd, after_strain, after_torsion)
        if ligand in ("l1", "l2", "l3"):
            c = "#26547C"
        else:
            c = "#FE6D73"
        ax.plot(
            [0, 1, 2, 3],
            [original_number, after_rmsd, after_strain, after_torsion],
            lw=2,
            c=c,
            marker="o",
            markersize=8,
            label=f"{name_conversion()[ligand]}-{after_torsion}",
        )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("num. conformers", fontsize=16)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["target", "RMSD", "strain", "torsion"])
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def simple_beeswarm(y, nbins=None, width=1.0):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    """
    y = np.asarray(y)
    if nbins is None:
        # nbins = len(y) // 6
        nbins = np.ceil(len(y) / 6).astype(int)

    # Get upper bounds of bins
    x = np.zeros(len(y))

    nn, ybins = np.histogram(y, bins=nbins)
    nmax = nn.max()

    # Divide indices into bins
    ibs = []  # np.nonzero((y>=ybins[0])*(y<=ybins[1]))[0]]
    for ymin, ymax in zip(ybins[:-1], ybins[1:]):
        i = np.nonzero((y > ymin) * (y <= ymax))[0]
        ibs.append(i)

    # Assign x indices
    dx = width / (nmax // 2)
    for i in ibs:
        yy = y[i]
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(yy)]
            a = i[j::2]
            b = i[j + 1 :: 2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def plot_all_ligand_pairings_simplified(
    results_dict,
    dihedral_cutoff,
    length_score_cutoff,
    angle_score_cutoff,
    strain_cutoff,
    outname,
):
    name = outname.replace(".png", "")
    logging.info(f"plotting: plot_all_ligand_pairings of {name}")

    fig, ax = plt.subplots(figsize=(8, 5))

    pair_to_x = {
        ("l1", "la"): {"x": 0, "a": [], "l": [], "g": []},
        ("l1", "lb"): {"x": 1, "a": [], "l": [], "g": []},
        ("l1", "lc"): {"x": 2, "a": [], "l": [], "g": []},
        ("l1", "ld"): {"x": 3, "a": [], "l": [], "g": []},
        ("l2", "la"): {"x": 4, "a": [], "l": [], "g": []},
        ("l2", "lb"): {"x": 5, "a": [], "l": [], "g": []},
        ("l2", "lc"): {"x": 6, "a": [], "l": [], "g": []},
        ("l2", "ld"): {"x": 7, "a": [], "l": [], "g": []},
        ("l3", "la"): {"x": 8, "a": [], "l": [], "g": []},
        ("l3", "lb"): {"x": 9, "a": [], "l": [], "g": []},
        ("l3", "lc"): {"x": 10, "a": [], "l": [], "g": []},
        ("l3", "ld"): {"x": 11, "a": [], "l": [], "g": []},
    }
    for pair_name in results_dict:
        small_l, large_l = pair_name.split(",")
        if "e" in small_l or "e" in large_l:
            continue

        for conf_pair in results_dict[pair_name]:
            rdict = results_dict[pair_name][conf_pair]

            if (
                abs(rdict["large_dihedral"]) > dihedral_cutoff
                or abs(rdict["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            pair_to_x[(small_l, large_l)]["a"].append(
                abs(rdict["angle_deviation"] - 1)
            )
            pair_to_x[(small_l, large_l)]["l"].append(
                abs(rdict["length_deviation"] - 1)
            )
            pair_to_x[(small_l, large_l)]["g"].append(rdict["geom_score"])

    width = 0.8
    bottom = np.zeros(12)
    for meas, c in zip(["a", "l", "g"], ["#083D77", "#F56476", "none"]):
        # for meas, c in zip(["a", "l"], ["#083D77", "#F56476"]):
        x = [pair_to_x[i]["x"] for i in pair_to_x]
        y = [np.mean(pair_to_x[i][meas]) for i in pair_to_x]
        # min_y = [min(pair_to_x[i][meas]) for i in pair_to_x]
        # max_y = [max(pair_to_x[i][meas]) for i in pair_to_x]

        if meas == "g":
            ec = "k"
            abottom = np.zeros(12)
            # error = None
        else:
            ec = "w"
            abottom = bottom
            # error = np.array((min_y, max_y))

        p = ax.bar(
            x,
            y,
            # c=c,
            # # s=20,
            # marker="o",
            # markersize=10,
            # lw=2,
            # alpha=1.0,
            width,
            # yerr=error,
            label=f"${meas}$",
            bottom=abottom,
            color=c,
            edgecolor=ec,
            linewidth=2,
        )
        if meas != "g":
            ax.bar_label(
                p, label_type="center", color="w", fontsize=12, fmt="%.2f"
            )
        # ax.fill_between(
        #     x,
        #     y1=min_y,
        #     y2=max_y,
        #     color=c,
        #     alpha=0.2,
        # )
        bottom += np.asarray(y)

    ax.tick_params(axis="both", which="major", labelsize=16)
    # ax.set_xlabel("$a$", fontsize=16)
    ax.set_ylabel("deviation", fontsize=16)
    # ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.6)
    # ax.axhline(y=1, c="gray", lw=2, linestyle="--")
    ax.axvline(x=3.5, c="gray", lw=2, linestyle="--")
    ax.axvline(x=7.5, c="gray", lw=2, linestyle="--")
    ax.text(x=1.2, y=1.5, s=name_conversion()["l1"], fontsize=16)
    ax.text(x=5.2, y=1.5, s=name_conversion()["l2"], fontsize=16)
    ax.text(x=9.2, y=1.5, s=name_conversion()["l3"], fontsize=16)

    ax.set_xticks(range(0, 12))
    ax.set_xticklabels(
        [f"{name_conversion()[i[1]]}" for i in pair_to_x],
        # rotation=20,
    )

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

def plot_all_ligand_pairings(
    results_dict,
    dihedral_cutoff,
    length_score_cutoff,
    angle_score_cutoff,
    strain_cutoff,
    outname,
):
    name = outname.replace(".png", "")
    logging.info(f"plotting: plot_all_ligand_pairings of {name}")

    fig, axs = plt.subplots(
        ncols=4,
        nrows=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    # large_to_ax = {
    #     "la": axs[0][0],
    #     "lb": axs[0][1],
    #     "lc": axs[1][0],
    #     "ld": axs[1][1],
    # }
    # small_to_c = {
    #     "l1": "#083D77",
    #     "l2": "#FFC15E",
    #     "l3": "#F56476",
    # }
    # small_to_x = {
    #     "l1": 0,
    #     "l2": 3,
    #     "l3": 6,
    # }

    # pair_to_x = {
    #     ("l1", "la"): 0,
    #     ("l1", "lb"): 1,
    #     ("l1", "lc"): 2,
    #     ("l1", "ld"): 3,
    #     ("l2", "la"): 4,
    #     ("l2", "lb"): 5,
    #     ("l2", "lc"): 6,
    #     ("l2", "ld"): 7,
    #     ("l3", "la"): 8,
    #     ("l3", "lb"): 9,
    #     ("l3", "lc"): 10,
    #     ("l3", "ld"): 11,
    # }

    for pair_name, ax in zip(results_dict, flat_axs):
        small_l, large_l = pair_name.split(",")
        if "e" in small_l or "e" in large_l:
            continue

        ax.set_title(
            f"{name_conversion()[small_l]}-{name_conversion()[large_l]}",
            fontsize=16,
        )
        all_as = []
        all_ls = []
        all_gs = []
        for conf_pair in results_dict[pair_name]:
            rdict = results_dict[pair_name][conf_pair]

            if (
                abs(rdict["large_dihedral"]) > dihedral_cutoff
                or abs(rdict["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            all_as.append(abs(rdict["angle_deviation"] - 1))
            all_ls.append(abs(rdict["length_deviation"] - 1))
            all_gs.append(rdict["geom_score"])

        print(pair_name, len(all_as), len(all_ls), len(all_gs))

        if len(all_as) != 0:
            ax.scatter(
                simple_beeswarm(all_as, width=0.4) + 0,
                all_as,
                c="#083D77",
                s=30,
                edgecolor="none",
                alpha=1.0,
            )
            ax.scatter(
                simple_beeswarm(all_ls, width=0.4) + 1,
                all_ls,
                c="#FFC15E",
                s=30,
                edgecolor="none",
                alpha=1.0,
            )
            ax.scatter(
                simple_beeswarm(all_gs, width=0.4) + 2,
                all_gs,
                c="#F56476",
                s=30,
                edgecolor="none",
                alpha=1.0,
            )
            # ax.boxplot(
            #     all_as,
            #     widths=0.4,
            #     showfliers=False,
            #     showcaps=False,
            #     positions=[0],
            # )
            # ax.boxplot(
            #     all_ls,
            #     widths=0.4,
            #     showfliers=False,
            #     showcaps=False,
            #     positions=[1],
            # )
            # ax.boxplot(
            #     all_gs,
            #     widths=0.4,
            #     showfliers=False,
            #     showcaps=False,
            #     positions=[2],
            # )

        ax.tick_params(axis="both", which="major", labelsize=16)
        # ax.set_xlabel("$a$", fontsize=16)
        ax.set_ylabel("deviation", fontsize=16)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1.6)
        # ax.axhline(y=1, c="gray", lw=2, linestyle="--")
        # ax.axvline(x=1, c="gray", lw=2, linestyle="--")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["$a$", "$l$", "$g$"])

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_ligand_pairings_2dhist(
    results_dict,
    dihedral_cutoff,
    length_score_cutoff,
    angle_score_cutoff,
    strain_cutoff,
    outname,
):
    name = outname.replace(".png", "")
    logging.info(f"plotting: plot_all_ligand_pairings of {name}")

    fig, axs = plt.subplots(
        ncols=4,
        nrows=3,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()

    xmin = ymin = 0
    xmax = ymax = 2

    for pair_name, ax in zip(results_dict, flat_axs):
        small_l, large_l = pair_name.split(",")
        if "e" in small_l or "e" in large_l:
            continue

        ax.set_title(
            f"{name_conversion()[small_l]}-{name_conversion()[large_l]}",
            fontsize=16,
        )
        all_as = []
        all_ls = []
        for conf_pair in results_dict[pair_name]:
            rdict = results_dict[pair_name][conf_pair]

            if (
                abs(rdict["large_dihedral"]) > dihedral_cutoff
                or abs(rdict["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            all_as.append(rdict["angle_deviation"])
            all_ls.append(rdict["length_deviation"])

        print(pair_name, len(all_as), len(all_ls))

        if len(all_ls) != 0:
            ax.scatter(
                all_as,
                all_ls,
                c="#083D77",
                alpha=0.2,
                s=50,
                edgecolor="k",
            )

    for ax in axs.flatten():
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("$a$", fontsize=16)
        ax.set_ylabel("$l$", fontsize=16)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axhline(y=1, c="gray", lw=2, linestyle="--")
        ax.axvline(x=1, c="gray", lw=2, linestyle="--")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
    raise SystemExit


def plot_ligand_pairing(results_dict, dihedral_cutoff, strain_cutoff, outname):
    name = outname.replace(".png", "")
    logging.info(f"plotting: plot_ligand_pairing of {name}")

    xmin = ymin = 0
    xmax = ymax = 2

    fig, ax = plt.subplots(figsize=(8, 5))

    all_xs = []
    all_ys = []
    num_points_total = 0
    num_points_tested = 0
    for pair_name in results_dict:
        rdict = results_dict[pair_name]
        num_points_total += 1

        if (
            abs(rdict["large_dihedral"]) > dihedral_cutoff
            or abs(rdict["small_dihedral"]) > dihedral_cutoff
        ):
            continue

        all_xs.append(rdict["angle_deviation"])
        all_ys.append(rdict["length_deviation"])

        num_points_tested += 1

    if len(all_ys) != 0:
        ax.scatter(
            all_xs,
            all_ys,
            c="#FFC15E",
            s=50,
            edgecolor="k",
        )

        name = name.split("_")[2] + "+" + name.split("_")[3]
        ax.set_title(
            (f"{name}: {num_points_tested} of " f"{num_points_total}"),
            fontsize=16,
        )

    ax.axhline(y=1, c="gray", lw=2, linestyle="--")
    ax.axvline(x=1, c="gray", lw=2, linestyle="--")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("$a$", fontsize=16)
    ax.set_ylabel("$l$", fontsize=16)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_single_distribution(
    results_dict,
    outname,
    yproperty,
):
    lab_prop = axes_labels(yproperty)

    fig, ax = plt.subplots(figsize=(8, 5))
    all_values = []
    for cid in results_dict:
        if results_dict[cid][yproperty] is None:
            continue

        if yproperty == "NN_BCN_angles":
            value = results_dict[cid][yproperty]["NN_BCN1"]
            all_values.append(value)
            value = results_dict[cid][yproperty]["NN_BCN2"]
            all_values.append(value)
        else:
            value = results_dict[cid][yproperty]
            all_values.append(value)

    if yproperty == "xtb_dmsoenergy":
        all_values = [i - min(all_values) for i in all_values]
        all_values = [i * 2625.5 for i in all_values]

    if lab_prop[1] is None:
        xbins = 50
    else:
        xwidth = 1
        xbins = np.arange(
            lab_prop[1][0] - xwidth,
            lab_prop[1][1] + xwidth,
            xwidth,
        )
    ax.hist(
        x=all_values,
        bins=xbins,
        density=False,
        histtype="stepfilled",
        stacked=True,
        linewidth=1.0,
        facecolor="#212738",
        alpha=1.0,
        edgecolor="k",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(lab_prop[0], fontsize=16)
    ax.set_ylabel("count", fontsize=16)
    ax.set_xlim(lab_prop[1])

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_vs_energy(results_dict, outname, yproperty):
    lab_prop = axes_labels(yproperty)

    fig, ax = plt.subplots(figsize=(8, 5))
    all_energies = []
    all_values = []
    for cid in results_dict:
        if results_dict[cid][yproperty] is None:
            continue

        energy = results_dict[cid]["xtb_dmsoenergy"]

        if yproperty == "NN_BCN_angles":
            value = results_dict[cid][yproperty]["NN_BCN1"]
            all_energies.append(energy)
            all_values.append(value)
            value = results_dict[cid][yproperty]["NN_BCN2"]
            all_energies.append(energy)
            all_values.append(value)
        else:
            value = results_dict[cid][yproperty]
            all_energies.append(energy)
            all_values.append(value)

    all_energies = [i - min(all_energies) for i in all_energies]
    all_energies = [i * 2625.5 for i in all_energies]

    ax.scatter(
        [i for i in all_values],
        [i for i in all_energies],
        c="k",
        s=50,
        alpha=1.0,
        edgecolors="k",
    )

    ax.axhline(y=5, lw=2, c="k", linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(lab_prop[0], fontsize=16)
    ax.set_ylabel("rel. xtb/DMSO energy [kJmol-1]", fontsize=16)
    ax.set_xlim(lab_prop[1])
    ax.set_ylim(0, 20)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_property(results_dict, outname, yproperty, ignore_topos=None):
    if ignore_topos is None:
        ignore_topos = ()

    cms = c_and_m_properties()
    lab_prop = axes_labels(yproperty)
    conv = lab_prop[2]

    fig, ax = plt.subplots(figsize=(16, 5))

    x_position = 0
    _x_names = []
    all_values = []
    for struct in results_dict:
        topo, l1, l2 = name_parser(struct)
        if topo in ignore_topos:
            continue
        s_values = results_dict[struct]
        if len(s_values) == 0:
            continue

        try:
            y = s_values[yproperty]
        except KeyError:
            if yproperty in (
                "avg_heli",
                "max_heli",
                "min_heli",
                "mm_distance",
                "pore_angle",
            ):
                y = -10
            else:
                y = s_values["pw_results"][yproperty]

        y = conv(y)
        x_position += 1
        c = cms[topo][0]
        all_values.append((x_position, y, c))
        if l2 is None:
            name = f"{topo} {l1}"
        else:
            name = f"{topo} {l1} {l2}"
        _x_names.append((x_position, name))

        ax.plot(
            [x_position, x_position],
            [0, y],
            c=c,
            lw=2,
        )

    ax.scatter(
        x=[i[0] for i in all_values],
        y=[i[1] for i in all_values],
        c=[i[2] for i in all_values],
        edgecolors="k",
        s=180,
        zorder=2,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("structure", fontsize=16)
    ax.set_ylabel(lab_prop[0], fontsize=16)

    # ax.set_xlim((0, 1))
    ax.set_ylim(lab_prop[1])
    ax.set_xticks([i[0] for i in _x_names])
    ax.set_xticklabels([i[1] for i in _x_names], rotation=90)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_qsqp(results_dict, outname, yproperty, ignore_topos=None):
    if ignore_topos is None:
        ignore_topos = ()

    cms = c_and_m_properties()
    lab_prop = axes_labels(yproperty)
    conv = lab_prop[2]

    to_plot = (
        "m6_l1",
        "m12_l2",
        "m24_l3",
        "m30_l3",
        "m2_la",
        "m2_lb",
        "m2_lc",
        "m2_ld",
        "m4_ls",
        "m2_ll1",
        "m2_ll2",
        "cis_l1_la",
        "cis_l1_lb",
        "cis_l1_lc",
        "cis_l1_ld",
        "cis_l2_la",
        "cis_l2_lb",
        "cis_l2_lc",
        "cis_l2_ld",
        "cis_l3_la",
        "cis_l3_lb",
        "cis_l3_lc",
        "cis_l3_ld",
        "cis_ll1_ls",
        "cis_ll2_ls",
    )

    fig, ax = plt.subplots(figsize=(16, 5))

    x_position = 0
    _x_names = []
    all_values = []
    for struct in results_dict:
        if struct not in to_plot:
            continue

        topo, l1, l2 = name_parser(struct)
        if topo in ignore_topos:
            continue
        s_values = results_dict[struct]
        if len(s_values) == 0:
            continue

        try:
            y = s_values[yproperty]
        except KeyError:
            if yproperty in (
                "avg_heli",
                "max_heli",
                "min_heli",
                "mm_distance",
                "pore_angle",
            ):
                y = -10
            else:
                y = s_values["pw_results"][yproperty]

        y = conv(y)
        x_position += 1
        c = cms[topo][0]
        all_values.append((x_position, y, c))
        if l2 is None:
            name = f"{topo} {l1}"
        else:
            name = f"{topo} {l1} {l2}"
        _x_names.append((x_position, name))

        ax.plot(
            [x_position, x_position],
            [0, y],
            c=c,
            lw=2,
        )

    ax.scatter(
        x=[i[0] for i in all_values],
        y=[i[1] for i in all_values],
        c=[i[2] for i in all_values],
        edgecolors="k",
        s=180,
        zorder=2,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("structure", fontsize=16)
    ax.set_ylabel(lab_prop[0], fontsize=16)

    # ax.set_xlim((0, 1))
    ax.set_ylim(0.90, 1.01)
    ax.set_xticks([i[0] for i in _x_names])
    ax.set_xticklabels([i[1] for i in _x_names], rotation=90)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def compare_cis_trans(results_dict, outname, yproperty):
    ignore_topos = ("m2", "m3", "m4", "m6", "m12", "m24", "m30")

    lab_prop = axes_labels(yproperty)
    conv = lab_prop[2]

    fig, ax = plt.subplots(figsize=(8, 5))

    _x_names = []
    _x_posis = []
    for struct in results_dict:
        topo, l1, l2 = name_parser(struct)
        name = f"{l1}+{l2}"
        if topo in ignore_topos:
            continue
        if name in _x_names:
            continue
        paired_topo = "cis" if topo == "trans" else "trans"
        paired_struct = f"{paired_topo}_{l1}_{l2}"

        cis_s = struct if "cis" in struct else paired_struct
        trans_s = paired_struct if "cis" in struct else struct
        # cis.
        s_values = results_dict[cis_s]
        try:
            y = s_values[yproperty]
        except KeyError:
            y = s_values["pw_results"][yproperty]

        cis = conv(y)

        # trans.
        s_values = results_dict[trans_s]
        try:
            y = s_values[yproperty]
        except KeyError:
            y = s_values["pw_results"][yproperty]

        trans = conv(y)

        if yproperty == "xtb_solv_opt_dmsoenergy":
            tplot = cis - trans
            tplot = tplot * 2625.5
        else:
            if cis is None or trans is None:
                tplot = None
            else:
                tplot = cis - trans

        xpos = len(_x_names)
        _x_names.append(name)
        _x_posis.append(xpos)
        ax.scatter(
            x=xpos,
            y=tplot,
            c="gold",
            edgecolors="k",
            s=180,
        )

    ax.axhline(y=0, lw=2, linestyle="--", c="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("cis-trans", fontsize=16)
    ax.set_title(lab_prop[0], fontsize=16)
    ax.set_xticks(_x_posis)
    ax.set_xticklabels(_x_names, rotation=45)
    ax.set_ylim(lab_prop[3])

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def method_c_map():
    methods = {
        "xtb_solv_opt_gasenergy_au": {
            "name": "GFN2-xTB",
            # "name": "x/g",
            # "long-name": "xTB/gas",
        },
        "xtb_solv_opt_dmsoenergy_au": {
            "name": "GFN2-xTB/DMSO",
            # "name": "x/d",
            # "long-name": "xtb/DMSO",
        },
        "pbe0_def2svp_sp_gas_kjmol": {
            # "name": "PBE0/def2-svp/GD3BJ/gas SP",
            "name": "d/g/s",
            # "long-name": "xTB/PBE0/def2-svp/GD3BJ/gas",
        },
        "pbe0_def2svp_sp_dmso_kjmol": {
            # "name": "PBE0/def2-svp/GD3BJ/DMSO SP",
            "name": "d/d/s",
            # "long-name": "xTB/PBE0/def2-svp/GD3BJ/DMSO",
        },
        "pbe0_def2svp_opt_gas_kjmol": {
            "name": "PBE0/def2-svp/GD3BJ",
            # "name": "d/g/o",
            # "long-name": "PBE0/PBE0/def2-svp/GD3BJ/gas",
        },
        "pbe0_def2svp_opt_dmso_kjmol": {
            "name": "PBE0/def2-svp/GD3BJ/DMSO",
            # "name": "d/d/o",
            # "long-name": "PBE0/PBE0/def2-svp/GD3BJ/DMSO",
        },
    }
    return methods


def plot_exchange_reactions(rxns, outname):
    _, _, l1, l2 = outname.split("_")

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = method_c_map()
    x_values = []
    y_values = []
    for i, method in enumerate(methods):
        method_rxns = rxns[method]
        r_string = (
            f"{method_rxns[0]['lhs_stoich']} * het. "
            "<->"
            f"{method_rxns[0]['l1_stoich']} * "
            f"{method_rxns[0]['l1_prefix'].upper()}({l1}) + "
            f"{method_rxns[0]['l2_stoich']} * "
            f"{method_rxns[0]['l2_prefix'].upper()}({l2}) "
        )

        for rxn in method_rxns:
            # l1_prefix = int(rxn['l1_prefix'][1])
            # l2_prefix = int(rxn['l2_prefix'][1])

            if rxn["lhs"] == 0 or rxn["rhs"] == 0:
                r_energy = 0
                ax.scatter(i, 0, c="r", marker="X", s=200, zorder=3)
            else:
                r_energy = float(rxn["lhs"]) - float(rxn["rhs"])

            r_energy = r_energy / int(rxn["lhs_stoich"])

            # ax.text(i-2, r_energy, r_string, fontsize=16)
            x_values.append((i, methods[method]["name"]))
            y_values.append(r_energy)

        ax.bar(
            x=[i[0] for i in x_values],
            height=[i for i in y_values],
            width=0.9,
            color="#212738",
            edgecolor="none",
            linewidth=1,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_title(r_string, fontsize=16)
    ax.set_xlabel("method", fontsize=16)
    ax.set_ylabel(
        "energy per het. cage [kJ mol-1]",
        fontsize=16,
    )
    ax.set_xticks([i[0] for i in x_values])
    ax.set_xticklabels([i[1] for i in x_values])
    # ax.legend(fontsize=16)
    # ax.set_xlim(lab_prop[1])
    ax.set_ylim(-100, 100)

    ax.axhline(y=0, lw=2, c="k", linestyle="--")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_exchange_reactions(all_rxns, outname):
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = method_c_map()

    x_shifts = (-0.3, 0, 0.3)
    width = 0.15

    x_positions = [i for i in range(len(all_rxns))]
    x_names = [
        name_conversion()[i.split("_")[1]]
        + " + "
        + name_conversion()[i.split("_")[2]]
        for i in all_rxns
    ]

    plotted = 0
    for method in methods:
        if "gas" in method:
            continue
        x_shift = x_shifts[plotted]
        plotted += 1
        method_y_values = []
        for hs in all_rxns:
            rxns = all_rxns[hs]
            method_rxn = rxns[method][0]
            if method_rxn["lhs"] == 0 or method_rxn["rhs"] == 0:
                r_energy = 0
            else:
                r_energy = float(method_rxn["lhs"]) - float(method_rxn["rhs"])
            r_energy = r_energy / int(method_rxn["lhs_stoich"])
            namings = hs.split("_")
            print(
                method,
                f"{namings[0]}-{name_conversion()[namings[1]]}-"
                f"{name_conversion()[namings[2]]}",
                r_energy,
            )
            method_y_values.append(r_energy)

        ax.bar(
            x=[i + x_shift for i in x_positions],
            height=[i for i in method_y_values],
            width=width,
            # color="#212738",
            edgecolor="none",
            linewidth=1,
            label=methods[method]["name"],
        )
    for xpos in x_positions[:-1]:
        ax.axvline(x=xpos + 0.5, lw=2, c="gray", linestyle="--")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(
        "energy per het. cage [kJ mol-1]",
        fontsize=16,
    )
    ax.set_xticks([i for i in x_positions])
    ax.set_xticklabels([i for i in x_names], rotation=45)
    ax.axhline(y=0, lw=2, c="k", linestyle="--")
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_exchange_reactions_production(all_rxns, outname):
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = method_c_map()

    x_shifts = (-0.25, 0.25)
    width = 0.3

    x_positions = [
        i for i, rs in enumerate(all_rxns) if rs in ("cis_l1_lb", "cis_l1_lc")
    ]
    x_names = [
        name_conversion()[i.split("_")[1]]
        + " + "
        + name_conversion()[i.split("_")[2]]
        for i in all_rxns
        if i in ("cis_l1_lb", "cis_l1_lc")
    ]

    plotted = 0
    for method in methods:
        if "gas" in method:
            continue
        if "sp" in method:
            continue
        x_shift = x_shifts[plotted]
        plotted += 1
        method_y_values = []
        for hs in all_rxns:
            if hs not in ("cis_l1_lb", "cis_l1_lc"):
                continue

            rxns = all_rxns[hs]
            method_rxn = rxns[method][0]
            r_energy = float(method_rxn["lhs"]) - float(method_rxn["rhs"])
            r_energy = r_energy / int(method_rxn["lhs_stoich"])
            method_y_values.append(r_energy)

        ax.bar(
            x=[i + x_shift for i in x_positions],
            height=[i for i in method_y_values],
            width=width,
            # color="#212738",
            edgecolor="none",
            linewidth=1,
            label=methods[method]["name"],
        )
    for xpos in x_positions[:-1]:
        ax.axvline(x=xpos + 0.5, lw=2, c="gray", linestyle="--")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(
        "energy per heteroleptic cage [kJ mol-1]",
        fontsize=16,
    )
    ax.set_xticks([i for i in x_positions])
    ax.set_xticklabels([i for i in x_names])  # , rotation=45)
    ax.set_xlim(0.5, 2.5)
    ax.axhline(y=0, lw=2, c="k", linestyle="--")
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.png"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def plot_homoleptic_exchange_reactions(rxns, outname):
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = method_c_map()
    for i, method in enumerate(methods):
        if "sp" in method:
            continue
        method_rxns = rxns[method]
        x_values = []
        y_values = []
        for i, rxn in enumerate(method_rxns):
            r_string = f"{rxn['l_prefix'].upper()}"
            if rxn["energy_per_stoich"] != 0:
                r_energy = float(rxn["energy_per_stoich"])
                x_values.append((i, r_string))
                y_values.append(r_energy)

        if "dmso" in method:
            linestyle = "-"
        else:
            linestyle = "--"
        ax.plot(
            [i[0] for i in x_values],
            [i - min(y_values) for i in y_values],
            linestyle=linestyle,
            marker="o",
            markersize=8,
            lw=3,
            label=methods[method]["name"],
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_title(f"{name_conversion()[rxn['l']]}", fontsize=16)
    ax.set_xlabel("homolepetic cage", fontsize=16)
    ax.set_ylabel(
        "relative energy / metal atom [kJ mol-1]",
        fontsize=16,
    )
    ax.set_ylim(0, 200)
    # ax.axhline(y=0, lw=2, c="k", linestyle="--")

    ax.set_xticks([i[0] for i in x_values])
    ax.set_xticklabels([i[1] for i in x_values])
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.png"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()
