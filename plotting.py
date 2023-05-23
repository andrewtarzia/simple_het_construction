#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for plotting functions.

Author: Andrew Tarzia

"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl

# from matplotlib.lines import Line2D
import logging
import numpy as np
import os

from env_set import figu_path
from utilities import name_parser


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
            "avg. strain energy [kJ mol-1]",
            (0, 50),
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
            "PBE0/def2-svp/D3?/gas SP energy [kJ mol-1]",
            (-400, -200),
            no_conv,
            None,
        ),
        "pbe0_def2svp_sp_dmso_kjmol": (
            "PBE0/def2-svp/D3?/DMSO SP energy [kJ mol-1]",
            (-400, -200),
            no_conv,
            None,
        ),
        "pbe0_def2svp_opt_gas_kjmol": (
            "PBE0/def2-svp/D3?/gas OPT energy [kJ mol-1]",
            (-400, -200),
            no_conv,
            None,
        ),
        "pbe0_def2svp_opt_dmso_kjmol": (
            "PBE0/def2-svp/D3?/DMSO OPT energy [kJ mol-1]",
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


def calculate_adev(a, c):
    # T1.
    return (2 * a + 2 * c) / 360


def calculate_ldev(a, A, B):
    # T2.
    bonding_vector_length = 2 * 2.05
    se = bonding_vector_length * np.sin(np.radians(a - 90))
    ideal_dist = A + 2 * se
    return B / ideal_dist


def calculate_strain(a, c, A, B):

    return abs(calculate_adev(a, c) - 1) + abs(
        calculate_ldev(a, A, B) - 1
    )


def plot_analytical_data_1():
    logging.info("plotting: plot_analytical_data_1")

    c = 60
    target_a = (360 - 2 * c) / 2
    A = 5
    target_e = 2 * 2.05 * np.sin(np.radians(target_a - 90))
    B = A + 2 * target_e

    a = np.linspace(90, 160, 100)

    fig, ax = plt.subplots(figsize=(8, 5))

    xs = []
    ys1 = []
    ys2 = []
    ys3 = []
    for aa in a:
        xs.append(aa)
        ys1.append(calculate_strain(aa, c, A, B))
        ys2.append(calculate_adev(aa, c))
        ys3.append(calculate_ldev(aa, A, B))

    ax.scatter(
        xs,
        ys1,
        label="strain",
    )
    ax.scatter(
        xs,
        ys2,
        label="angle deviation",
    )
    ax.scatter(
        xs,
        ys3,
        label="length deviation",
    )
    ax.axvline(x=target_a, c="gray", linestyle="--")
    ax.axhline(y=1, c="gray", linestyle="--")

    ax.legend(fontsize=16)
    ax.set_ylim(0, None)
    ax.set_xlabel("a values", fontsize=16)
    ax.set_ylabel("strain ; deviation", fontsize=16)
    ax.set_title(
        f"effect of mismatch in a: (c:{c}, A:{A}, B:{round(B, 2)})"
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), "alytl_1.png"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_analytical_data_2():
    logging.info("plotting: plot_analytical_data_2")

    c = 50
    a = (360 - 2 * c) / 2
    A = 8
    e = 2 * 2.05 * np.sin(np.radians(a - 90))
    target_B = A + 2 * e

    B = np.linspace(1, 20, 100)

    fig, ax = plt.subplots(figsize=(8, 5))

    xs = []
    ys1 = []
    ys2 = []
    ys3 = []
    for BB in B:
        xs.append(BB)
        ys1.append(calculate_strain(a, c, A, BB))
        ys2.append(calculate_adev(a, c))
        ys3.append(calculate_ldev(a, A, BB))

    ax.scatter(
        xs,
        ys1,
        label="strain",
    )
    ax.scatter(
        xs,
        ys2,
        label="angle deviation",
    )
    ax.scatter(
        xs,
        ys3,
        label="length deviation",
    )
    ax.axvline(x=target_B, c="gray", linestyle="--")
    ax.axhline(y=1, c="gray", linestyle="--")

    ax.legend(fontsize=16)
    ax.set_ylim(0, None)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("B values", fontsize=16)
    ax.set_ylabel("strain ; deviation", fontsize=16)
    ax.set_title(f"effect of mismatch in B: (a: {a}, c:{c}, A:{A})")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), "alytl_2.png"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_analytical_data_3():
    logging.info("plotting: plot_analytical_data_3")

    c = 50
    A = 8

    a = np.linspace(90, 160, 50)
    B = np.linspace(1, 20, 50)

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))

    xs = []
    ys = []
    lds = []
    ads = []
    cs = []
    for BB in B:
        for aa in a:
            xs.append(BB)
            ys.append(aa)
            lds.append(calculate_ldev(aa, A, BB))
            ads.append(calculate_adev(aa, c))
            cs.append(calculate_strain(aa, c, A, BB))

    vmin = 0
    vmax = 0.5
    axs[0].scatter(
        xs,
        ys,
        c=cs,
        vmin=vmin,
        vmax=vmax,
        cmap="Blues_r",
    )
    axs[1].scatter(
        lds,
        ads,
        c=cs,
        vmin=vmin,
        vmax=vmax,
        cmap="Blues_r",
    )
    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("strain", fontsize=16)

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("B values", fontsize=16)
    axs[0].set_ylabel("a values", fontsize=16)
    axs[1].set_xlabel("length deviation", fontsize=16)
    axs[1].set_ylabel("angle deviation", fontsize=16)
    axs[0].set_title(f"effect of mismatch in B, a: (c:{c}, A:{A})")
    axs[1].set_title(f"effect of mismatch in B, a: (c:{c}, A:{A})")
    axs[1].set_xlim(0, 2)
    axs[1].set_ylim(0, 2)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), "alytl_3.png"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_geom_scores_vs_dihedral_cutoff(
    results_dict,
    geom_score_cutoff,
    outname,
):
    logging.info("plotting: plot_geom_scores_vs_dihedral_cutoff")

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))

    xs = np.linspace(0.2, 180, 60)

    for pair_name in results_dict:
        rdict = results_dict[pair_name]
        if pair_name not in (
            "e3,e2",
            "e16,e17",
            "e16,e14",
            "e10,e17",
            "e16,e10",
            "e1,e4",
            "e11,e10",
            "e11,e14",
            "e12,e14",
            "e11,e13",
            "e12,e13",
            "e15,e14",
            "e18,e10",
            "e18,e14",
        ):
            continue

        ys = []
        ys2 = []
        for dihedral_threshold in xs:
            all_scores = 0
            less_scores = 0
            for cid_pair in rdict:
                all_scores += 1

                if (
                    abs(rdict[cid_pair]["large_dihedral"])
                    > dihedral_threshold
                    or abs(rdict[cid_pair]["small_dihedral"])
                    > dihedral_threshold
                ):
                    continue

                geom_score = rdict[cid_pair]["geom_score"]

                if geom_score < geom_score_cutoff:
                    less_scores += 1

            if all_scores == 0:
                ys.append(0)
                ys2.append(0)
            else:
                ys.append((less_scores / all_scores) * 100)
                ys2.append(all_scores)

        axs[0].plot(
            xs,
            ys,
            lw=3,
            label=pair_name,
        )
        axs[1].plot(
            xs,
            ys2,
            lw=3,
            # label=pair_name,
        )

    axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("dihedral threshold", fontsize=16)
    axs[0].set_ylabel(r"% good", fontsize=16)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("dihedral threshold", fontsize=16)
    axs[1].set_ylabel("num. pairs", fontsize=16)

    fig.legend(ncol=4, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_geom_scores_vs_max_strain(
    results_dict,
    dihedral_cutoff,
    geom_score_cutoff,
    outname,
):
    logging.info("plotting: plot_geom_scores_vs_max_strain")
    raise NotImplementedError(
        "no longer needed - strain is removed elsewhere"
    )

    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))

    xs = np.linspace(0.1, 50, 20)

    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if pair_name not in (
            "e3,e2",
            "e16,e17",
            "e16,e14",
            "e10,e17",
            "e16,e10",
            "e1,e4",
            "e11,e10",
            "e11,e14",
            "e12,e14",
            "e11,e13",
            "e12,e13",
            "e15,e14",
            "e18,e10",
            "e18,e14",
        ):
            continue

        ys = []
        ys2 = []
        for max_strain in xs:
            all_scores = 0
            less_scores = 0
            for cid_pair in rdict:
                all_scores += 1

                if (
                    abs(rdict[cid_pair]["large_dihedral"])
                    > dihedral_cutoff
                    or abs(rdict[cid_pair]["small_dihedral"])
                    > dihedral_cutoff
                ):
                    continue

                geom_score = rdict[cid_pair]["geom_score"]

                if geom_score < geom_score_cutoff:
                    less_scores += 1

            if all_scores == 0:
                ys.append(0)
                ys2.append(0)
            else:
                ys.append((less_scores / all_scores) * 100)
                ys2.append(all_scores)

        axs[0].plot(
            xs,
            ys,
            lw=3,
            label=pair_name,
        )
        axs[1].plot(
            xs,
            ys2,
            lw=3,
            # label=pair_name,
        )

    axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("strain threshold [kJ/mol]", fontsize=16)
    axs[0].set_ylabel(r"% good", fontsize=16)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("strain threshold [kJ/mol]", fontsize=16)
    axs[1].set_ylabel("num. pairs", fontsize=16)

    fig.legend(ncol=4, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_geom_scores_vs_threshold(
    results_dict,
    dihedral_cutoff,
    strain_cutoff,
    outname,
    experimental_ligand_outcomes,
):
    logging.info("plotting: plot_geom_scores_vs_threshold")

    fig, axs = plt.subplots(nrows=3, sharey=True, figsize=(8, 12))

    xs = np.linspace(0, 1, 30)
    colour_map = {
        "yes": "#086788",
        "no": "#F9A03F",
        "tested": "#F9A03F",
    }

    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            ename = pair_name.split(",")
            if (ename[0], ename[1]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[
                    (ename[0], ename[1])
                ]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[
                    (ename[1], ename[0])
                ]
            else:
                continue
            colour = colour_map[edata]
        else:
            continue
            colour = colour_map["tested"]

        ys = []
        ys_l = []
        ys_a = []
        for threshold in xs:
            all_scores = 0
            less_scores = 0
            less_l_scores = 0
            less_a_scores = 0
            for cid_pair in rdict:
                all_scores += 1

                if (
                    abs(rdict[cid_pair]["large_dihedral"])
                    > dihedral_cutoff
                    or abs(rdict[cid_pair]["small_dihedral"])
                    > dihedral_cutoff
                ):
                    continue

                geom_score = rdict[cid_pair]["geom_score"]
                if geom_score < threshold:
                    less_scores += 1

                angle_score = rdict[cid_pair]["angle_deviation"]
                if abs(angle_score - 1) < threshold:
                    less_a_scores += 1

                length_score = rdict[cid_pair]["length_deviation"]
                if abs(length_score - 1) < threshold:
                    less_l_scores += 1

            if all_scores == 0:
                ys.append(0)
            else:
                ys.append((less_scores / all_scores) * 100)

            if all_scores == 0:
                ys_l.append(0)
            else:
                ys_l.append((less_l_scores / all_scores) * 100)

            if all_scores == 0:
                ys_a.append(0)
            else:
                ys_a.append((less_a_scores / all_scores) * 100)

        axs[0].plot(
            xs,
            ys,
            lw=2,
            c=colour,
            # label=pair_name,
        )
        axs[1].plot(
            xs,
            ys_l,
            lw=2,
            c=colour,
        )
        axs[2].plot(
            xs,
            ys_a,
            lw=2,
            c=colour,
        )

    # axs[0].set_ylim(0, 101)
    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("geom score threshold", fontsize=16)
    axs[0].set_ylabel(r"% good", fontsize=16)

    # axs[1].set_ylim(0, 101)
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("length score threshold", fontsize=16)
    axs[1].set_ylabel(r"% good", fontsize=16)

    # axs[2].set_ylim(0, 101)
    axs[2].tick_params(axis="both", which="major", labelsize=16)
    axs[2].set_xlabel("angle score threshold", fontsize=16)
    axs[2].set_ylabel(r"% good", fontsize=16)

    fig.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_geom_scores_density(
    results_dict,
    outname,
    dihedral_cutoff,
    geom_score_cutoff,
    strain_cutoff,
    experimental_ligand_outcomes,
):
    logging.info("plotting: plot_all_geom_scores_density")

    colour_map = {
        "yes": "#086788",
        "no": "white",
        "tested": "#F9A03F",
    }

    fig, ax = plt.subplots(figsize=(16, 5))

    categories_gs = {}
    colour_list = []

    crosses = {}
    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            ename = pair_name.split(",")
            if (ename[0], ename[1]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[
                    (ename[0], ename[1])
                ]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[
                    (ename[1], ename[0])
                ]
            else:
                continue
            colour_list.append(colour_map[edata])
        else:
            colour_list.append(colour_map["tested"])

        if len(rdict) == 0:
            crosses[pair_name] = (colour_list[-1], "X")
            categories_gs[pair_name] = 0
            continue

        all_scores = 0
        less_scores = 0
        for cid_pair in rdict:
            all_scores += 1

            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"])
                > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            if geom_score < geom_score_cutoff:
                less_scores += 1

        categories_gs[pair_name] = (less_scores / all_scores) * 100

    ax.set_title(
        (
            f"dihedral cut {dihedral_cutoff}; "
            f"strain cut {strain_cutoff} kj/mol"
        ),
        fontsize=16,
    )
    ax.bar(
        categories_gs.keys(),
        categories_gs.values(),
        color=colour_list,
        edgecolor="k",
        lw=2,
    )

    for x, tstr in enumerate(categories_gs):
        if tstr in crosses:
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

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(r"% good geometry score", fontsize=16)
    ax.set_xticks(range(len(categories_gs)))
    ax.set_xticklabels(categories_gs.keys(), rotation=45)

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

    ax.legend(handles=legend_elements, ncol=5, fontsize=16)

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
                edata = experimental_ligand_outcomes[
                    (ename[0], ename[1])
                ]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[
                    (ename[1], ename[0])
                ]
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
                or abs(rdict[cid_pair]["small_dihedral"])
                > dihedral_cutoff
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
    geom_score_cutoff,
    angle_score_cutoff,
    length_score_cutoff,
    strain_cutoff,
    experimental_ligand_outcomes,
):

    logging.info("plotting: plot_all_geom_scores_categorical")

    colour_map = {
        "yes": "#086788",
        "no": "white",
        "tested": "#F9A03F",
    }

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    flat_axs = axs.flatten()

    categories_gcs = []
    categories_gs = []
    categories_ls = []
    categories_as = []
    colour_list = []
    x_list = []

    crosses = {}
    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            ename = pair_name.split(",")
            if (ename[0], ename[1]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[
                    (ename[0], ename[1])
                ]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[
                    (ename[1], ename[0])
                ]
            else:
                continue
            colour_list.append(colour_map[edata])
            if edata == "yes":
                x_list.append(2)
            elif edata == "no":
                x_list.append(3)
        else:
            colour_list.append(colour_map["tested"])
            x_list.append(1)

        if len(rdict) == 0:
            crosses[pair_name] = (colour_list[-1], "X")
            categories_gs.append(-1)
            categories_gcs.append(-1)
            categories_ls.append(-1)
            categories_as.append(-1)
            continue

        all_scores = 0
        less_scores = 0
        less_a_scores = 0
        less_l_scores = 0
        min_geom_score = 1e24
        geom_scores = []
        for cid_pair in rdict:
            all_scores += 1

            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"])
                > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            geom_scores.append(geom_score)
            if geom_score < min_geom_score:
                min_geom_score = geom_score

            if geom_score < geom_score_cutoff:
                less_scores += 1

            angle_score = rdict[cid_pair]["angle_deviation"]
            if abs(angle_score - 1) < angle_score_cutoff:
                less_a_scores += 1

            length_score = rdict[cid_pair]["length_deviation"]
            if abs(length_score - 1) < length_score_cutoff:
                less_l_scores += 1

        if all_scores == 0 or min_geom_score == 1e24:
            categories_gs.append(0)
            categories_gcs.append(1)
            categories_as.append(0)
            categories_ls.append(0)
        else:
            # categories_gcs.append(min_geom_score)
            # categories_gcs.append(np.median(geom_scores))
            deviations = np.array(geom_scores)
            categories_gcs.append(
                np.sqrt(
                    np.sum(deviations * deviations) / len(geom_scores)
                )
            )

            categories_gs.append((less_scores / all_scores) * 100)
            categories_as.append((less_a_scores / all_scores) * 100)
            categories_ls.append((less_l_scores / all_scores) * 100)

        flat_axs[0].text(
            x_list[-1] - 0.5,
            categories_gcs[-1],
            s=pair_name,
        )
        flat_axs[1].text(
            x_list[-1] - 0.5,
            categories_gs[-1],
            s=pair_name,
        )
        # flat_axs[2].text(
        #     x_list[-1] - 0.5,
        #     categories_ls[-1],
        #     s=pair_name,
        # )
        # flat_axs[3].text(
        #     x_list[-1] - 0.5,
        #     categories_as[-1],
        #     s=pair_name,
        # )

    flat_axs[0].scatter(
        [i + np.random.uniform(-1, 1, 1) * 0.3 for i in x_list],
        categories_gcs,
        color=colour_list,
        edgecolor="k",
        s=120,
    )
    flat_axs[0].axvline(x=1.5, c="gray", linestyle="--")
    flat_axs[0].axvline(x=2.5, c="gray", linestyle="--")
    flat_axs[0].tick_params(axis="both", which="major", labelsize=16)
    # flat_axs[0].set_ylabel("min geometry score", fontsize=16)
    # flat_axs[0].set_ylabel("median geometry score", fontsize=16)
    flat_axs[0].set_ylabel("RMSD geometry score", fontsize=16)
    flat_axs[0].set_xlim(0.5, 3.5)
    flat_axs[0].set_xticks((1, 2, 3))
    flat_axs[0].set_xticklabels(("this work", "yes", "no"))
    flat_axs[0].set_ylim(0, None)

    flat_axs[1].scatter(
        [i + np.random.uniform(-1, 1, 1) * 0.3 for i in x_list],
        categories_gs,
        color=colour_list,
        edgecolor="k",
        s=120,
    )
    flat_axs[1].axvline(x=1.5, c="gray", linestyle="--")
    flat_axs[1].axvline(x=2.5, c="gray", linestyle="--")
    flat_axs[1].tick_params(axis="both", which="major", labelsize=16)
    flat_axs[1].set_ylabel(r"% good geometry score", fontsize=16)
    flat_axs[1].set_xlim(0.5, 3.5)
    flat_axs[1].set_xticks((1, 2, 3))
    flat_axs[1].set_xticklabels(("this work", "yes", "no"))
    flat_axs[1].set_ylim(0, None)

    # flat_axs[2].scatter(
    #     [i + np.random.uniform(-1, 1, 1) * 0.3 for i in x_list],
    #     categories_ls,
    #     color=colour_list,
    #     edgecolor="k",
    #     s=120,
    # )
    # flat_axs[2].axvline(x=1.5, c="gray", linestyle="--")
    # flat_axs[2].axvline(x=2.5, c="gray", linestyle="--")
    # flat_axs[2].tick_params(axis="both", which="major", labelsize=16)
    # flat_axs[2].set_ylabel(r"% good length score", fontsize=16)
    # flat_axs[2].set_xlim(0.5, 3.5)
    # flat_axs[2].set_xticks((1, 2, 3))
    # flat_axs[2].set_xticklabels(("this work", "yes", "no"))
    # flat_axs[2].set_ylim(0, None)

    # flat_axs[3].scatter(
    #     [i + np.random.uniform(-1, 1, 1) * 0.3 for i in x_list],
    #     categories_as,
    #     color=colour_list,
    #     edgecolor="k",
    #     s=120,
    # )
    # flat_axs[3].axvline(x=1.5, c="gray", linestyle="--")
    # flat_axs[3].axvline(x=2.5, c="gray", linestyle="--")
    # flat_axs[3].tick_params(axis="both", which="major", labelsize=16)
    # flat_axs[3].set_ylabel(r"% good angle score", fontsize=16)
    # flat_axs[3].set_xlim(0.5, 3.5)
    # flat_axs[3].set_xticks((1, 2, 3))
    # flat_axs[3].set_xticklabels(("this work", "yes", "no"))
    # flat_axs[3].set_ylim(0, None)

    # legend_elements = []
    # for i in colour_map:
    #     legend_elements.append(
    #         Patch(
    #             facecolor=colour_map[i],
    #             label=i,
    #             alpha=1.0,
    #             edgecolor="k",
    #         ),
    #     )

    # ax.legend(handles=legend_elements, ncol=5, fontsize=16)

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
                edata = experimental_ligand_outcomes[
                    (ename[0], ename[1])
                ]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[
                    (ename[1], ename[0])
                ]
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
                or abs(rdict[cid_pair]["small_dihedral"])
                > dihedral_cutoff
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


def gs_table(
    results_dict,
    dihedral_cutoff,
    geom_score_cutoff,
    strain_cutoff,
):
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
                or abs(rdict[cid_pair]["small_dihedral"])
                > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            geom_scores.append(geom_score)
            if geom_score < min_geom_score:
                min_geom_score = geom_score

            if geom_score < geom_score_cutoff:
                good_geoms += 1

        logging.info(
            (
                f"{pair_name}: {round(min_geom_score, 2)}, "
                f"{round((good_geoms/total_tested)*100, 0)} "
                f"{round(np.mean(geom_scores), 3)} "
                f"({round(np.std(geom_scores), 3)}) "
            )
        )


def plot_conformer_props(
    structure_results,
    outname,
    dihedral_cutoff,
    strain_cutoff,
    experimental_ligand_outcomes,
    low_energy_values,
):
    logging.info("plotting: plot_conformer_props")

    fig, ax = plt.subplots(figsize=(16, 5))

    categories_num_confs = {}
    categories_num_strain = {}
    categories_num_dihedral = {}

    for ligand in structure_results:
        sres = structure_results[ligand]
        low_energy = low_energy_values[ligand][1]

        num_confs = 0
        num_strain = 0
        num_dihedral = 0

        for cid in sres:
            num_confs += 1
            strain = sres[cid]["UFFEnergy;kj/mol"] - low_energy
            dihedral = sres[cid]["NCCN_dihedral"]

            if strain > strain_cutoff:
                continue
            num_strain += 1

            if abs(dihedral) > dihedral_cutoff:
                continue
            num_dihedral += 1

        categories_num_confs[ligand] = num_confs
        categories_num_strain[ligand] = num_strain
        categories_num_dihedral[ligand] = num_dihedral

    p = ax.bar(
        categories_num_dihedral.keys(),
        categories_num_dihedral.values(),
        color="gold",
        edgecolor="k",
        lw=2,
    )
    ax.bar_label(
        p,
        labels=[
            round((i / j) * 100, 1)
            for i, j in zip(
                categories_num_dihedral.values(),
                categories_num_confs.values(),
            )
        ],
        label_type="center",
        padding=3,
        fontsize=16,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("num. conformers", fontsize=16)

    ax.set_xticks(range(len(categories_num_confs)))
    ax.set_xticklabels(categories_num_confs.keys(), rotation=45)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_geom_scores(
    results_dict,
    dihedral_cutoff,
    geom_score_cutoff,
    outname,
):
    logging.info("plotting: plot_geom_scores")

    fig, ax = plt.subplots(figsize=(8, 5))

    all_xs = []
    all_ys = []
    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        all_xs.append(rdict["geom_score"])
        all_ys.append(
            max(
                abs(rdict["large_dihedral"]),
                abs(rdict["small_dihedral"]),
            )
        )

    if len(all_ys) != 0:

        hb = ax.hexbin(
            [i for i in all_xs],
            [i for i in all_ys],
            gridsize=40,
            # extent=(-2, 2, -2, 2),
            cmap="inferno",
            bins="log",
        )
        fig.colorbar(hb, ax=ax, label="log10(N)")

    ax.axvline(x=geom_score_cutoff, lw=2, c="gray", linestyle="--")
    ax.axhline(y=dihedral_cutoff, lw=2, c="gray", linestyle="--")

    # ax.scatter(
    #     [i for i in all_xs],
    #     [i for i in all_ys],
    #     c="teal",
    #     s=30,
    #     alpha=1.0,
    #     edgecolors="none",
    # )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("geom score", fontsize=16)
    ax.set_ylabel("max dihedral", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_ligand_pairing(
    results_dict,
    dihedral_cutoff,
    geom_score_cutoff,
    length_score_cutoff,
    angle_score_cutoff,
    strain_cutoff,
    outname,
):
    name = outname.replace(".png", "")
    logging.info(f"plotting: plot_ligand_pairing of {name}")

    xmin = ymin = 0
    xmax = ymax = 2

    fig, ax = plt.subplots(figsize=(8, 5))

    all_xs = []
    all_ys = []
    num_points_total = 0
    num_good = 0
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

        if rdict["geom_score"] < geom_score_cutoff:
            num_good += 1

    if len(all_ys) != 0:
        hb = ax.hexbin(
            [i for i in all_xs],
            [i for i in all_ys],
            gridsize=40,
            extent=(xmin, xmax, ymin, ymax),
            cmap="inferno",
            bins="log",
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_title("log10(N)", fontsize=16)

        name = name.split("_")[1] + "+" + name.split("_")[2]
        ax.set_title(
            (
                f"{name}: {num_good} good of {num_points_tested} of "
                f"{num_points_total}"
            ),
            fontsize=16,
        )

    # ax.plot(
    #     (
    #         1 - geom_score_cutoff,
    #         1,
    #         1 + geom_score_cutoff,
    #         1,
    #         1 - geom_score_cutoff,
    #     ),
    #     (1, 1 + geom_score_cutoff, 1, 1 - geom_score_cutoff, 1),
    #     linestyle="--",
    #     linewidth=2,
    #     c="k",
    # )

    ax.axhline(y=1, c="gray", lw=2, linestyle="--")
    ax.axvline(x=1, c="gray", lw=2, linestyle="--")

    # ax.axvline(
    #     x=1 - angle_score_cutoff, c="gray", lw=0.5, linestyle="--"
    # )
    # ax.axvline(
    #     x=1 + angle_score_cutoff, c="gray", lw=0.5, linestyle="--"
    # )

    # ax.axhline(
    #     y=1 - length_score_cutoff, c="gray", lw=0.5, linestyle="--"
    # )
    # ax.axhline(
    #     y=1 + length_score_cutoff, c="gray", lw=0.5, linestyle="--"
    # )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("angle deviation []", fontsize=16)
    ax.set_ylabel("length deviation []", fontsize=16)
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


def plot_vs_energy(
    results_dict,
    outname,
    yproperty,
):

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


def plot_pair_distribution(
    results_dict,
    outname,
    yproperty,
):
    raise NotImplementedError("not using this for this paper, I think.")
    lab_prop = axes_labels(yproperty)

    fig, ax = plt.subplots(figsize=(8, 5))
    all_xvalues = []
    # all_yvalues = []
    for cid in results_dict:
        dictionary = results_dict[cid][yproperty]
        xvalue = dictionary[""]

        all_xvalues.append(xvalue)
        # all_yvalues.append(yvalue)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(lab_prop[0], fontsize=16)
    ax.set_ylabel(lab_prop[0], fontsize=16)

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
            # "name": "xtb gas",
            "name": "x/g",
        },
        "xtb_solv_opt_dmsoenergy_au": {
            # "name": "xtb DMSO",
            "name": "x/d",
        },
        "pbe0_def2svp_sp_gas_kjmol": {
            # "name": "PBE0/def2-svp/D3?/gas SP",
            "name": "d/g/s",
        },
        "pbe0_def2svp_sp_dmso_kjmol": {
            # "name": "PBE0/def2-svp/D3?/DMSO SP",
            "name": "d/d/s",
        },
        "pbe0_def2svp_opt_gas_kjmol": {
            # "name": "PBE0/def2-svp/D3?/gas OPT",
            "name": "d/g/o",
        },
        "pbe0_def2svp_opt_dmso_kjmol": {
            # "name": "PBE0/def2-svp/D3?/DMSO OPT",
            "name": "d/d/o",
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


def plot_homoleptic_exchange_reactions(rxns, outname):

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = method_c_map()
    for i, method in enumerate(methods):
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
    ax.set_title(f"{rxn['l']}", fontsize=16)
    ax.set_xlabel("reaction", fontsize=16)
    ax.set_ylabel(
        "rel. energy / metal atom [kJ mol-1]",
        fontsize=16,
    )
    ax.axhline(y=0, lw=2, c="k", linestyle="--")

    ax.set_xticks([i[0] for i in x_values])
    ax.set_xticklabels([i[1] for i in x_values])
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
