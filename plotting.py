#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for plotting functions.

Author: Andrew Tarzia

"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib

import logging
import numpy as np
import os

from env_set import figu_path
from utilities import name_parser, name_conversion
from run_ligand_analysis import vector_length


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


def plot_all_geom_scores_simplified(
    results_dict,
    dihedral_cutoff,
    outname,
    experimental_ligand_outcomes,
):
    name = outname.replace(".png", "")
    logging.info(f"plotting: plot_all_geom_scores_simplified of {name}")

    fig, ax = plt.subplots(figsize=(16, 5))

    pair_to_x = {
        # Forms.
        # ("l1", "lb"): {"x": 1, "a": [], "l": [], "g": []},
        # ("l1", "lc"): {"x": 2, "a": [], "l": [], "g": []},
        ("e16", "e10"): {"x": 1, "a": [], "l": [], "g": []},
        ("e16", "e17"): {"x": 2, "a": [], "l": [], "g": []},
        ("e16", "e14"): {"x": 3, "a": [], "l": [], "g": []},
        ("e11", "e10"): {"x": 4, "a": [], "l": [], "g": []},
        ("e11", "e13"): {"x": 5, "a": [], "l": [], "g": []},
        ("e11", "e14"): {"x": 6, "a": [], "l": [], "g": []},
        ("e18", "e10"): {"x": 7, "a": [], "l": [], "g": []},
        ("e18", "e14"): {"x": 8, "a": [], "l": [], "g": []},
        ("e12", "e10"): {"x": 9, "a": [], "l": [], "g": []},
        ("e12", "e13"): {"x": 10, "a": [], "l": [], "g": []},
        ("e12", "e14"): {"x": 11, "a": [], "l": [], "g": []},
        # Mixture.
        ("e13", "e14"): {"x": 12, "a": [], "l": [], "g": []},
        # Our work, does not form.
        # ("l1", "la"): {"x": 15, "a": [], "l": [], "g": []},
        # ("l1", "ld"): {"x": 16, "a": [], "l": [], "g": []},
        # ("l2", "la"): {"x": 17, "a": [], "l": [], "g": []},
        # ("l2", "lb"): {"x": 18, "a": [], "l": [], "g": []},
        # ("l2", "lc"): {"x": 19, "a": [], "l": [], "g": []},
        # ("l2", "ld"): {"x": 20, "a": [], "l": [], "g": []},
        # ("l3", "la"): {"x": 21, "a": [], "l": [], "g": []},
        # ("l3", "lb"): {"x": 22, "a": [], "l": [], "g": []},
        # ("l3", "lc"): {"x": 23, "a": [], "l": [], "g": []},
        # ("l3", "ld"): {"x": 24, "a": [], "l": [], "g": []},
        # Narcissitic Self sort.
        ("e3", "e2"): {"x": 13, "a": [], "l": [], "g": []},
        # Does not form - different heteroleptic.
        ("e10", "e17"): {"x": 14, "a": [], "l": [], "g": []},
        ("e1", "e3"): {"x": 15, "a": [], "l": [], "g": []},
        ("e1", "e4"): {"x": 16, "a": [], "l": [], "g": []},
        ("e1", "e6"): {"x": 17, "a": [], "l": [], "g": []},
        ("e3", "e9"): {"x": 18, "a": [], "l": [], "g": []},
    }
    for pair_name in results_dict:
        small_l, large_l = pair_name.split(",")
        if "e" not in small_l:
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
    bottom = np.zeros(len(pair_to_x))
    for meas, c in zip(["a", "l", "g"], ["#083D77", "#F56476", "none"]):
        # for meas, c in zip(["a", "l"], ["#083D77", "#F56476"]):
        x = [pair_to_x[i]["x"] for i in pair_to_x]
        y = [np.mean(pair_to_x[i][meas]) for i in pair_to_x]
        # min_y = [min(pair_to_x[i][meas]) for i in pair_to_x]
        # max_y = [max(pair_to_x[i][meas]) for i in pair_to_x]

        if meas == "g":
            ec = "k"
            abottom = np.zeros(len(pair_to_x))
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
        else:
            ax.bar_label(
                p,
                label_type="edge",
                padding=0.1,
                color="k",
                fontsize=12,
                fmt="%.2f",
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
    ax.axvline(x=11.5, c="gray", lw=2, linestyle="--")
    ax.axvline(x=13.5, c="gray", lw=2, linestyle="--")

    ax.set_xticks(range(1, len(pair_to_x) + 1))
    ax.set_xticklabels(
        [
            f"{name_conversion()[i[0]]}-{name_conversion()[i[1]]}"
            for i in pair_to_x
        ],
        rotation=45,
    )

    ax.legend(fontsize=16, ncol=3)

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
    experimental_ligand_outcomes,
):
    logging.info("plotting: plot_all_geom_scores_categorical")

    colour_map = {
        "previous work": "#086788",
        # "experimental": "white",
        "this work": "#F9A03F",
    }

    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(16, 5))
    flat_axs = axs.flatten()

    categories_gcs = []
    categories_gs = []
    categories_as = []
    categories_ls = []
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
            categories_as.append(-1)
            categories_ls.append(-1)
            continue

        all_scores = 0
        geom_scores = []
        a_scores = []
        l_scores = []
        for cid_pair in rdict:
            all_scores += 1

            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            geom_scores.append(geom_score)
            l_score = abs(rdict[cid_pair]["length_deviation"] - 1)
            l_scores.append(l_score)
            a_score = abs(rdict[cid_pair]["angle_deviation"] - 1)
            a_scores.append(a_score)

        if all_scores == 0:
            categories_gs.append(0)
            categories_gcs.append(1)
            categories_as.append(0)
            categories_ls.append(1)
        else:
            categories_gs.append(min(geom_scores))
            categories_gcs.append(np.mean(geom_scores))
            categories_as.append(np.mean(a_scores))
            categories_ls.append(np.mean(l_scores))

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

    flat_axs[2].scatter(
        [i + np.random.uniform(-1, 1, 1) * 0.3 for i in x_list],
        categories_as,
        color=colour_list,
        edgecolor="k",
        s=120,
    )
    flat_axs[2].axvline(x=1.5, c="gray", linestyle="--")
    flat_axs[2].axvline(x=2.5, c="gray", linestyle="--")
    flat_axs[2].tick_params(axis="both", which="major", labelsize=16)
    flat_axs[2].set_ylabel(r"$a_{\mathrm{avg}}$", fontsize=16)
    flat_axs[2].set_xlim(0.5, 2.5)
    flat_axs[2].set_xticks((1, 2))
    flat_axs[2].set_xticklabels(("forms", "does not form"))
    flat_axs[2].set_ylim(0, 1.5)

    flat_axs[3].scatter(
        [i + np.random.uniform(-1, 1, 1) * 0.3 for i in x_list],
        categories_ls,
        color=colour_list,
        edgecolor="k",
        s=120,
    )
    flat_axs[3].axvline(x=1.5, c="gray", linestyle="--")
    flat_axs[3].axvline(x=2.5, c="gray", linestyle="--")
    flat_axs[3].tick_params(axis="both", which="major", labelsize=16)
    flat_axs[3].set_ylabel(r"$l_{\mathrm{avg}}$", fontsize=16)
    flat_axs[3].set_xlim(0.5, 2.5)
    flat_axs[3].set_xticks((1, 2))
    flat_axs[3].set_xticklabels(("forms", "does not form"))
    flat_axs[3].set_ylim(0, 1.5)

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


def gs_table(results_dict, dihedral_cutoff):
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
    show_cbar=True,
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
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

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


def gs_table_plot(results_dict, dihedral_cutoff, prefix):
    logging.info("plotting: gs table")

    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(16, 3))
    flat_axs = axs.flatten()

    matrix1 = np.zeros(shape=(3, 4))
    matrix2 = np.zeros(shape=(3, 4))
    matrix3 = np.zeros(shape=(3, 4))
    matrix4 = np.zeros(shape=(3, 4))
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

        total_tested = 0
        geom_scores = []
        l_scores = []
        a_scores = []
        for cid_pair in rdict:
            total_tested += 1

            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            geom_scores.append(geom_score)
            l_score = abs(rdict[cid_pair]["length_deviation"] - 1)
            l_scores.append(l_score)
            a_score = abs(rdict[cid_pair]["angle_deviation"] - 1)
            a_scores.append(a_score)

        matrix1[m_i][m_j] = min(geom_scores)
        matrix2[m_i][m_j] = np.mean(geom_scores)
        matrix3[m_i][m_j] = np.mean(a_scores)
        matrix4[m_i][m_j] = np.mean(l_scores)

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
    vmax = 1.0
    im, cbar = heatmap(
        matrix1,
        rows,
        cols,
        ax=flat_axs[0],
        cmap="magma_r",
        cbarlabel=r"$g_{\mathrm{min}}$",
        vmin=0,
        vmax=vmax,
        show_cbar=False,
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
        vmax=vmax,
        show_cbar=False,
    )
    _ = annotate_heatmap(im, valfmt="{x:.2f}")
    im, cbar = heatmap(
        matrix3,
        rows,
        cols,
        ax=flat_axs[2],
        cmap="magma_r",
        cbarlabel=r"$a_{\mathrm{avg}}$",
        vmin=0,
        vmax=vmax,
        show_cbar=False,
    )
    _ = annotate_heatmap(im, valfmt="{x:.2f}")
    im, cbar = heatmap(
        matrix4,
        rows,
        cols,
        ax=flat_axs[3],
        cmap="magma_r",
        cbarlabel=r"$l_{\mathrm{avg}}$",
        vmin=0,
        vmax=vmax,
        show_cbar=False,
    )
    _ = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{prefix}_g_tables.png"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8, 3))
    flat_axs = axs.flatten()
    im, cbar = heatmap(
        matrix4,
        rows,
        cols,
        ax=flat_axs[0],
        cmap="magma_r",
        cbarlabel=r"$l_{\mathrm{avg}}$",
        vmin=0,
        vmax=vmax,
        show_cbar=True,
    )
    im, cbar = heatmap(
        matrix4,
        rows,
        cols,
        ax=flat_axs[1],
        cmap="magma_r",
        cbarlabel=r"$l_{\mathrm{avg}}$",
        vmin=0,
        vmax=vmax,
        show_cbar=True,
    )
    _ = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{prefix}_g_tables_cbar.pdf"),
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
        else:
            ax.bar_label(
                p,
                label_type="edge",
                padding=0.1,
                color="k",
                fontsize=12,
                fmt="%.2f",
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

    xmin = 0.6
    ymin = 0.6
    xmax = 2.0
    ymax = 2.0

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

        print(
            pair_name,
            len(all_as),
            len(all_ls),
            round(max(all_as), 1),
            round(min(all_as), 1),
            round(max(all_ls), 1),
            round(min(all_ls), 1),
        )

        if len(all_ls) != 0:
            ax.scatter(
                all_as,
                all_ls,
                c="#FFC20A",
                alpha=0.2,
                s=70,
                edgecolor="none",
            )
            ax.scatter(
                np.mean(all_as),
                np.mean(all_ls),
                c="r",
                alpha=1,
                s=70,
                marker="X",
                edgecolor="k",
            )

    for ax in axs.flatten():
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("$a$", fontsize=16)
        ax.set_ylabel("$l$", fontsize=16)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axhline(y=1, c="gray", lw=1, linestyle="--")
        ax.axvline(x=1, c="gray", lw=1, linestyle="--")
        ax.set_xticks([0.75, 1, 1.25, 1.5, 1.75])
        ax.set_yticks([0.75, 1, 1.25, 1.5, 1.75])

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_ligand_pairings_conformers(
    results_dict,
    structure_results,
    dihedral_cutoff,
    outname,
):
    logging.info("plotting: plot_all_ligand_pairings_conformers")

    fig, axs = plt.subplots(
        ncols=12,
        nrows=1,
        sharex=True,
        sharey=True,
        figsize=(16, 4),
    )
    flat_axs = axs.flatten()

    lhs_anchor = (-1.5, 0)
    rhs_anchor = (1.5, 0)

    for pair_name, ax in zip(results_dict, flat_axs):
        small_l, large_l = pair_name.split(",")
        if "e" in small_l or "e" in large_l:
            continue

        ax.set_title(
            f"{name_conversion()[small_l]}-{name_conversion()[large_l]}",
            fontsize=16,
        )
        l_confs = {}
        s_confs = {}
        for conf_pair in results_dict[pair_name]:
            rdict = results_dict[pair_name][conf_pair]
            small_l, large_l = pair_name.split(",")
            small_cid, large_cid = conf_pair.split(",")
            if (
                abs(rdict["large_dihedral"]) > dihedral_cutoff
                or abs(rdict["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            large_c_dict = structure_results[large_l][large_cid]
            small_c_dict = structure_results[small_l][small_cid]
            pd_length = 2.05

            # 180 - angle, to make it the angle toward the binding interaction.
            # E.g. To become internal angle of trapezoid.
            s_angle1 = np.radians(
                (180 - small_c_dict["NN_BCN_angles"]["NN_BCN1"]) - 90
            )
            s_angle2 = np.radians(
                (180 - small_c_dict["NN_BCN_angles"]["NN_BCN2"]) - 90
            )
            l_angle1 = np.radians(
                90 - (180 - large_c_dict["NN_BCN_angles"]["NN_BCN1"])
            )
            l_angle2 = np.radians(
                90 - (180 - large_c_dict["NN_BCN_angles"]["NN_BCN2"])
            )
            l_confs[large_cid] = {
                "N1": (
                    lhs_anchor[0],
                    lhs_anchor[1] + large_c_dict["NN_distance"] / 2,
                ),
                "N2": (
                    lhs_anchor[0],
                    lhs_anchor[1] - large_c_dict["NN_distance"] / 2,
                ),
                "B1": (
                    lhs_anchor[0] + (pd_length * np.cos(l_angle1)),
                    (lhs_anchor[1] + large_c_dict["NN_distance"] / 2)
                    - (pd_length * np.sin(l_angle1)),
                ),
                "B2": (
                    lhs_anchor[0] + (pd_length * np.cos(l_angle2)),
                    (lhs_anchor[1] - large_c_dict["NN_distance"] / 2)
                    + (pd_length * np.sin(l_angle2)),
                ),
            }
            s_confs[large_cid] = {
                "N1": (
                    rhs_anchor[0],
                    rhs_anchor[1] + small_c_dict["NN_distance"] / 2,
                ),
                "N2": (
                    rhs_anchor[0],
                    rhs_anchor[1] - small_c_dict["NN_distance"] / 2,
                ),
                "B1": (
                    rhs_anchor[0] - (pd_length * np.cos(s_angle1)),
                    (
                        rhs_anchor[1]
                        + small_c_dict["NN_distance"] / 2
                        + (pd_length * np.sin(s_angle1))
                    ),
                ),
                "B2": (
                    rhs_anchor[0] - (pd_length * np.cos(s_angle2)),
                    (rhs_anchor[1] - small_c_dict["NN_distance"] / 2)
                    - (pd_length * np.sin(s_angle2)),
                ),
            }

        print(large_l, l_confs.keys())
        print(small_l, s_confs.keys())
        for cid in l_confs:
            ax.plot(
                (l_confs[cid]["N1"][0], l_confs[cid]["N2"][0]),
                (l_confs[cid]["N1"][1], l_confs[cid]["N2"][1]),
                c="#374A67",
                alpha=0.2,
                # s=100,
                lw=3,
                marker="o",
                markersize=6,
                markeredgecolor="none",
            )
            ax.plot(
                (l_confs[cid]["N1"][0], l_confs[cid]["B1"][0]),
                (l_confs[cid]["N1"][1], l_confs[cid]["B1"][1]),
                c="#374A67",
                alpha=0.2,
                # s=100,
                lw=3,
            )
            ax.plot(
                (l_confs[cid]["N2"][0], l_confs[cid]["B2"][0]),
                (l_confs[cid]["N2"][1], l_confs[cid]["B2"][1]),
                c="#374A67",
                alpha=0.2,
                # s=100,
                lw=3,
            )
            # ax.scatter(
            #     (l_confs[cid]["B1"][0], l_confs[cid]["B2"][0]),
            #     (l_confs[cid]["B1"][1], l_confs[cid]["B2"][1]),
            #     c="gold",
            #     alpha=0.2,
            #     s=100,
            #     marker="X",
            #     edgecolor="none",
            # )

        for cid in s_confs:
            ax.plot(
                (s_confs[cid]["N1"][0], s_confs[cid]["N2"][0]),
                (s_confs[cid]["N1"][1], s_confs[cid]["N2"][1]),
                c="#F0803C",
                alpha=0.2,
                # s=100,
                lw=3,
                marker="o",
                markersize=6,
                markeredgecolor="none",
            )
            ax.plot(
                (s_confs[cid]["N1"][0], s_confs[cid]["B1"][0]),
                (s_confs[cid]["N1"][1], s_confs[cid]["B1"][1]),
                c="#F0803C",
                alpha=0.2,
                # s=100,
                lw=3,
            )
            ax.plot(
                (s_confs[cid]["N2"][0], s_confs[cid]["B2"][0]),
                (s_confs[cid]["N2"][1], s_confs[cid]["B2"][1]),
                c="#F0803C",
                alpha=0.2,
                # s=100,
                lw=3,
            )
            # ax.scatter(
            #     (s_confs[cid]["B1"][0], s_confs[cid]["B2"][0]),
            #     (s_confs[cid]["B1"][1], s_confs[cid]["B2"][1]),
            #     c="gold",
            #     alpha=0.2,
            #     s=100,
            #     marker="X",
            #     edgecolor="none",
            # )

    for ax in axs.flatten():
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.axis("off")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-15, 15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(
            lhs_anchor[0],
            lhs_anchor[1],
            c="k",
            alpha=1.0,
            s=120,
            edgecolor="w",
            zorder=4,
        )
        ax.scatter(
            rhs_anchor[0],
            rhs_anchor[1],
            c="k",
            alpha=1.0,
            s=120,
            edgecolor="white",
            zorder=4,
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_ligand_pairings_2dhist_fig5(
    results_dict, dihedral_cutoff, outname
):
    name = outname.replace(".png", "")
    logging.info(f"plotting: plot_all_ligand_pairings of {name}")

    fig, axs = plt.subplots(
        ncols=3,
        nrows=1,
        sharex=True,
        sharey=True,
        figsize=(8, 3),
    )
    flat_axs = axs.flatten()

    xmin = 0.6
    ymin = 0.6
    xmax = 1.6
    ymax = 1.6

    targets = (
        "l1,lb",
        "l2,lb",
        "l1,la",
    )

    for pair_name, ax in zip(targets, flat_axs):
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

        print(
            pair_name,
            len(all_as),
            len(all_ls),
            round(max(all_as), 1),
            round(min(all_as), 1),
            round(max(all_ls), 1),
            round(min(all_ls), 1),
        )

        if len(all_ls) != 0:
            ax.scatter(
                all_as,
                all_ls,
                c="#FFC20A",
                alpha=0.2,
                s=70,
                edgecolor="none",
            )
            ax.scatter(
                np.mean(all_as),
                np.mean(all_ls),
                c="r",
                alpha=1,
                s=70,
                marker="X",
                edgecolor="k",
            )

    for ax in axs.flatten():
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axhline(y=1, c="gray", lw=1, linestyle="--")
        ax.axvline(x=1, c="gray", lw=1, linestyle="--")
        ax.set_xticks([0.75, 1, 1.25, 1.5])
        ax.set_yticks([0.75, 1, 1.25, 1.5])
    axs[1].set_xlabel("$a$", fontsize=16)
    axs[0].set_ylabel("$l$", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_ligand_pairing(results_dict, dihedral_cutoff, outname):
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


def plot_topo_energy(results_dict, outname):
    fig, ax = plt.subplots(figsize=(8, 5))

    to_plot = {
        "l1": {
            "systems": (
                "m2_l1",
                "m3_l1",
                "m4_l1",
                "m6_l1",
                "m12_l1",
                "m24_l1",
            ),
        },
        "l2": {
            "systems": (
                "m2_l2",
                "m3_l2",
                "m4_l2",
                "m6_l2",
                "m12_l2",
                "m24_l2",
            ),
        },
        "l3": {
            "systems": (
                "m2_l3",
                "m3_l3",
                "m4_l3",
                "m6_l3",
                "m12_l3",
                "m24_l3",
                "m30_l3",
            ),
        },
    }

    for ligand in to_plot:
        x_position = 0
        _x_names = []
        dmso_values = []
        for struct in to_plot[ligand]["systems"]:
            topo, l1, l2 = name_parser(struct)
            try:
                s_values = results_dict[struct]
            except KeyError:
                continue

            if len(s_values) == 0:
                continue
            x_position += 1

            name = topo.upper()
            num_metals = int(name.split("M")[-1])
            _x_names.append((x_position, name))

            dmso_energy = s_values["xtb_solv_opt_dmsoenergy_au"]
            dmso_values.append((x_position, dmso_energy, num_metals))

        print(dmso_values)
        min_dmso = min([i[1] / i[2] for i in dmso_values])
        print(min_dmso)

        dmso_values = [
            (i[0], ((i[1] / i[2]) - min_dmso) * 2625.5, i[2])
            for i in dmso_values
        ]
        print(dmso_values)

        ax.plot(
            [i[0] for i in dmso_values],
            [i[1] for i in dmso_values],
            # markersize=8,
            # marker="o",
            lw=2,
            label=name_conversion()[l1],
        )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_ylabel("total energy / Pd [kJmol$^{-1}$]", fontsize=16)

        # ax.set_xlim((0, 1))
        ax.set_ylim(0.0, None)
        ax.set_xticks([i[0] for i in _x_names])
        ax.set_xticklabels([i[1] for i in _x_names], rotation=0)

    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_pdntest(results_dict, dihedral_cutoff, outname):
    name = outname.replace(".png", "")
    logging.info(f"plotting: plot_pdntest of {name}")

    pair_to_c = {
        ("l1", "lb"): "#083D77",
        ("l2", "lb"): "#FFC15E",
        ("l3", "lb"): "#F56476",
    }

    fig, axs = plt.subplots(
        ncols=2,
        nrows=1,
        sharex=True,
        sharey=True,
        figsize=(16, 5),
    )

    for pair in pair_to_c:
        small_l, large_l = pair
        pair_name = ",".join(pair)
        if "e" in small_l or "e" in large_l:
            continue

        all_ls = {}
        all_as = {}
        for pdn in results_dict:
            all_ls[pdn] = []
            all_as[pdn] = []
            rdict = results_dict[pdn]
            for conf_pair in rdict[pair_name]:
                cdict = rdict[pair_name][conf_pair]
                if (
                    abs(cdict["large_dihedral"]) > dihedral_cutoff
                    or abs(cdict["small_dihedral"]) > dihedral_cutoff
                ):
                    continue

                all_as[pdn].append(abs(cdict["angle_deviation"] - 1))
                all_ls[pdn].append(abs(cdict["length_deviation"] - 1))

        axs[0].plot(
            [i for i in all_as],
            [np.mean(all_as[i]) for i in all_as],
            lw=3,
            marker="o",
            markersize=6,
            c=pair_to_c[pair],
            label=f"{name_conversion()[pair[0]]}-{name_conversion()[pair[1]]}",
        )
        axs[1].plot(
            [i for i in all_ls],
            [np.mean(all_ls[i]) for i in all_ls],
            lw=3,
            marker="o",
            markersize=6,
            c=pair_to_c[pair],
            label=f"{name_conversion()[pair[0]]}-{name_conversion()[pair[1]]}",
        )

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel(r"Pd-N distance [$\mathrm{\AA}$]", fontsize=16)
    axs[0].set_ylabel("mean $a$", fontsize=16)
    axs[0].axvline(x=vector_length(), c="gray", lw=2, linestyle="--")
    axs[0].legend(fontsize=16)
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel(r"Pd-N distance [$\mathrm{\AA}$]", fontsize=16)
    axs[1].set_ylabel("mean $l$", fontsize=16)
    axs[1].axvline(x=vector_length(), c="gray", lw=2, linestyle="--")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
