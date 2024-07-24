#!/usr/bin/env python
# Distributed under the terms of the MIT License.

"""Module for plotting functions.

Author: Andrew Tarzia

"""

import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from env_set import figu_path
from run_ligand_analysis import vector_length
from utilities import expt_name_conversion, name_conversion, name_parser


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

    def avg_strain_to_kjmol(y):
        return np.mean(list(y.values())) * 2625.5

    return {
        "min_order_param": ("min op.", (0, 1), no_conv),
        "pore_diameter_opt": (
            r"pyWindow pore diameter [$\mathrm{\AA}$]",
            (0, 10),
            no_conv,
            None,
        ),
        "xtb_dmsoenergy": (
            "xtb/DMSO energy [kJmol-1]",
            (0, 20),
            no_conv,
            None,
        ),
        "xtb_gsolv_au": (
            "xtb/DMSO G_solv/Pd [kJ mol-1]",
            (None, None),
            no_conv,
            (-20, 20),
        ),
        "xtb_gsasa_au": (
            "xtb/DMSO G_sasa/Pd [kJ mol-1]",
            (None, None),
            no_conv,
            (-20, 20),
        ),
        "xtb_lig_strain_au": (
            "avg. xtb/DMSO strain energy [kJ mol-1]",
            (5, 15),
            avg_strain_to_kjmol,
            (-20, 20),
        ),
        "xtb_sasa": (
            r"total SASA [$\mathrm{\AA}^{2}$]",
            (1500, 2200),
            no_conv,
            (0, 20),
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
        ("e16", "e10"): {"x": 1, "a": [], "l": [], "g": [], "success": True},
        ("e16", "e17"): {"x": 2, "a": [], "l": [], "g": [], "success": True},
        ("e10", "e17"): {"x": 3, "a": [], "l": [], "g": [], "success": False},
        ("e11", "e10"): {"x": 4, "a": [], "l": [], "g": [], "success": True},
        ("e16", "e14"): {"x": 5, "a": [], "l": [], "g": [], "success": True},
        ("e18", "e14"): {"x": 6, "a": [], "l": [], "g": [], "success": True},
        ("e18", "e10"): {"x": 7, "a": [], "l": [], "g": [], "success": True},
        ("e12", "e10"): {"x": 8, "a": [], "l": [], "g": [], "success": True},
        ("e11", "e14"): {"x": 9, "a": [], "l": [], "g": [], "success": True},
        ("e12", "e14"): {"x": 10, "a": [], "l": [], "g": [], "success": True},
        ("e11", "e13"): {"x": 11, "a": [], "l": [], "g": [], "success": True},
        ("e12", "e13"): {"x": 12, "a": [], "l": [], "g": [], "success": True},
        # Mixture.
        ("e13", "e14"): {"x": 13, "a": [], "l": [], "g": [], "success": False},
        ("e11", "e12"): {"x": 14, "a": [], "l": [], "g": [], "success": False},
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
        # ("e3", "e2"): {"x": 13, "a": [], "l": [], "g": []},
        # # Does not form - different heteroleptic.
        # ("e1", "e3"): {"x": 15, "a": [], "l": [], "g": []},
        # ("e1", "e4"): {"x": 16, "a": [], "l": [], "g": []},
        # ("e1", "e6"): {"x": 17, "a": [], "l": [], "g": []},
        # ("e3", "e9"): {"x": 18, "a": [], "l": [], "g": []},
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
    for meas, c in zip(
        ["a", "l", "g"], ["#083D77", "#F56476", "none"], strict=False
    ):
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
                p, label_type="center", color="w", fontsize=16, fmt="%.2f"
            )
        else:
            ax.bar_label(
                p,
                label_type="edge",
                padding=0.1,
                color="k",
                fontsize=16,
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
    ax.set_ylim(0, 0.9)
    # ax.axhline(y=1, c="gray", lw=2, linestyle="--")
    # ax.axvline(x=11.5, c="gray", lw=2, linestyle="--")
    # ax.axvline(x=12.5, c="gray", lw=2, linestyle="--")

    for i in pair_to_x:
        if not pair_to_x[i]["success"]:
            ax.scatter(
                pair_to_x[i]["x"],
                0.73,
                c="r",
                marker="X",
                edgecolor="k",
                s=150,
            )

    ax.set_xticks([pair_to_x[i]["x"] for i in pair_to_x])
    ax.set_xticklabels([expt_name_conversion(i[0], i[1]) for i in pair_to_x])

    ax.legend(fontsize=16, ncol=3)

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
            f"{pair_name}: {round(min_geom_score, 2)}, "
            f"{round((good_geoms/total_tested)*100, 0)} "
            f"{round(np.mean(geom_scores), 2)} "
            f"({round(np.std(geom_scores), 2)}) "
        )


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
        print(ligand, original_number, after_rmsd, after_strain, after_torsion)
        if ligand in ("l1", "l2", "l3"):
            if ligand == "l1":
                # label = f"{name_conversion()[ligand]}: {after_torsion}"
                label = r"1$^{\mathrm{X}}$"
            else:
                label = None
            c = "#26547C"
        else:
            c = "#FE6D73"
            if ligand == "la":
                label = r"2$^{\mathrm{X}}$"
            else:
                label = None
        ax.plot(
            [0, 1, 2, 3],
            [original_number, after_rmsd, after_strain, after_torsion],
            lw=2,
            c=c,
            marker="o",
            markersize=8,
            label=label,
        )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("screening stage", fontsize=16)
    ax.set_ylabel("number conformers", fontsize=16)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["initial", "RMSD", "strain", "torsion"])
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def simple_beeswarm(y, nbins=None, width=1.0):
    """Returns x coordinates for the points in ``y``, so that plotting ``x`` and
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
    for ymin, ymax in zip(ybins[:-1], ybins[1:], strict=False):
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
    for meas, c in zip(
        ["a", "l", "g"], ["#083D77", "#F56476", "none"], strict=False
    ):
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

    ax.set_xticks(range(12))
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

    for pair_name, ax in zip(results_dict, flat_axs, strict=False):
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

        ax.tick_params(axis="both", which="major", labelsize=16)
        # ax.set_xlabel("$a$", fontsize=16)
        ax.set_ylabel("deviation", fontsize=16)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1.6)
        # ax.axhline(y=1, c="gray", lw=2, linestyle="--")
        # ax.axvline(x=1, c="gray", lw=2, linestyle="--")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["|$a-1$|", "|$l-1$|", "$g$"])

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

    for pair_name, ax in zip(results_dict, flat_axs, strict=False):
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

    for pair_name, ax in zip(results_dict, flat_axs, strict=False):
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

    for pair_name, ax in zip(targets, flat_axs, strict=False):
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


def plot_strain_pore_sasa(results_dict, outname):
    xlab_prop = axes_labels("pore_diameter_opt")
    ylab_prop = axes_labels("xtb_sasa")
    clab_prop = axes_labels("xtb_lig_strain_au")

    fig, ax = plt.subplots(figsize=(8, 5))

    to_plot = (
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
    )

    xs = []
    ys = []
    cs = []
    forms_x = []
    forms_y = []
    for struct in to_plot:
        topo, l1, l2 = name_parser(struct)

        s_values = results_dict[struct]
        if len(s_values) == 0:
            continue
        xs.append(s_values["pw_results"]["pore_diameter_opt"])
        ys.append(s_values["xtb_sasa"]["total_sasa/A2"])
        cs.append(
            np.mean(list(s_values["xtb_lig_strain_au"].values())) * 2625.5
        )
        if struct in ("cis_l1_lb", "cis_l1_lc"):
            forms_x.append(s_values["pw_results"]["pore_diameter_opt"])
            forms_y.append(s_values["xtb_sasa"]["total_sasa/A2"])

    ax.scatter(
        x=forms_x,
        y=forms_y,
        c="r",
        edgecolors="none",
        s=300,
    )
    ax.scatter(
        x=xs,
        y=ys,
        c=cs,
        edgecolors="k",
        s=180,
        vmin=clab_prop[1][0],
        vmax=clab_prop[1][1],
        cmap="Blues",
    )

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=clab_prop[1][0], vmax=clab_prop[1][1])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(clab_prop[0], fontsize=16)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xlab_prop[0], fontsize=16)
    ax.set_ylabel(ylab_prop[0], fontsize=16)

    ax.set_xlim(xlab_prop[1])
    ax.set_ylim(ylab_prop[1])

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_strain(results_dict, outname, yproperty):
    cms = c_and_m_properties()
    lab_prop = axes_labels(yproperty)
    conv = lab_prop[2]

    fig, ax = plt.subplots(figsize=(8, 5))

    to_plot = (
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
    )

    x_position = 0
    _x_names = []
    all_values = []
    for struct in to_plot:
        topo, l1, l2 = name_parser(struct)

        s_values = results_dict[struct]
        if len(s_values) == 0:
            continue
        y = s_values[yproperty]
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


def plot_sasa(results_dict, outname, yproperty):
    cms = c_and_m_properties()
    lab_prop = axes_labels(yproperty)

    fig, ax = plt.subplots(figsize=(8, 5))

    to_plot = (
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
    )

    x_position = 0
    _x_names = []
    all_values = []
    for struct in to_plot:
        topo, l1, l2 = name_parser(struct)

        s_values = results_dict[struct]
        if len(s_values) == 0:
            continue
        y = s_values[yproperty]["total_sasa/A2"]

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


def plot_pore(results_dict, outname, yproperty):
    cms = c_and_m_properties()
    lab_prop = axes_labels(yproperty)

    fig, ax = plt.subplots(figsize=(8, 5))

    to_plot = (
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
    )

    x_position = 0
    _x_names = []
    all_values = []
    for struct in to_plot:
        topo, l1, l2 = name_parser(struct)

        s_values = results_dict[struct]
        if len(s_values) == 0:
            continue
        y = s_values["pw_results"][yproperty]

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


def plot_topo_energy(results_dict, outname, solvent=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    if solvent is None:
        s_key = "xtb_solv_opt_dmsoenergy_au"
    elif solvent == "gas":
        s_key = "xtb_solv_opt_gasenergy_au"

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
            "c": "tab:blue",
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
            "c": "tab:orange",
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
            "c": "tab:green",
        },
    }

    for ligand in to_plot:
        x_position = 0
        _x_names = []
        values = []
        fey_values = []
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

            energy = s_values[s_key]
            values.append((x_position, energy, num_metals))
            if "xtb_solv_opt_dmsofreeenergy_au" in s_values and ligand == "l1":
                freeenergy = s_values["xtb_solv_opt_dmsofreeenergy_au"]
                fey_values.append((x_position, freeenergy, num_metals))

        min_value = min([i[1] / i[2] for i in values])
        values = [
            (i[0], ((i[1] / i[2]) - min_value) * 2625.5, i[2]) for i in values
        ]

        ax.plot(
            [i[0] for i in values],
            [i[1] for i in values],
            markersize=12,
            marker="o",
            markerfacecolor=to_plot[ligand]["c"],
            markeredgecolor="k",
            lw=2,
            c=to_plot[ligand]["c"],
            label=name_conversion()[l1],
        )

        if len(fey_values) != 0:
            print(fey_values)
            min_fey_value = min([i[1] / i[2] for i in fey_values])
            fey_values = [
                (i[0], ((i[1] / i[2]) - min_fey_value) * 2625.5, i[2])
                for i in fey_values
            ]
            print(min_fey_value)
            print(fey_values)
            ax.plot(
                [i[0] for i in fey_values],
                [i[1] for i in fey_values],
                markersize=12,
                marker="s",
                markerfacecolor=to_plot[ligand]["c"],
                markeredgecolor="k",
                lw=2,
                ls="--",
                c=to_plot[ligand]["c"],
                label=f"{name_conversion()[l1]}: FE",
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
        os.path.join(figu_path(), f"{outname}.png"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def plot_gsolv(results_dict, outname, yproperty):
    cms = c_and_m_properties()
    lab_prop = axes_labels(yproperty)
    conv = lab_prop[2]

    fig, ax = plt.subplots(figsize=(8, 5))

    to_plot = (
        "m2_la",
        "m2_lb",
        "m2_lc",
        "m2_ld",
        "m6_l1",
        "cis_l1_la",
        "cis_l1_lb",
        "cis_l1_lc",
        "cis_l1_ld",
        "m12_l2",
        "cis_l2_la",
        "cis_l2_lb",
        "cis_l2_lc",
        "cis_l2_ld",
        "m30_l3",
        "cis_l3_la",
        "cis_l3_lb",
        "cis_l3_lc",
        "cis_l3_ld",
    )
    x_position = 0
    _x_names = []
    all_values = []
    for struct in to_plot:
        topo, l1, l2 = name_parser(struct)

        s_values = results_dict[struct]
        if len(s_values) == 0:
            continue
        y = s_values[yproperty]
        y = conv(y)
        x_position += 1
        c = cms[topo][0]
        if l2 is None:
            name = f"{topo} {l1}"
        else:
            name = f"{topo} {l1} {l2}"
        if "cis" in name:
            num_metals = 2
        else:
            num_metals = int(topo.split("m")[-1])
        y = (y / num_metals) * 2625.5

        _x_names.append((x_position, name))
        # input()
        all_values.append((x_position, y, c))
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


def plot_gsasa(results_dict, outname, yproperty):
    cms = c_and_m_properties()
    lab_prop = axes_labels(yproperty)
    conv = lab_prop[2]

    fig, ax = plt.subplots(figsize=(8, 5))

    to_plot = (
        "m2_la",
        "m2_lb",
        "m2_lc",
        "m2_ld",
        "m6_l1",
        "cis_l1_la",
        "cis_l1_lb",
        "cis_l1_lc",
        "cis_l1_ld",
        "m12_l2",
        "cis_l2_la",
        "cis_l2_lb",
        "cis_l2_lc",
        "cis_l2_ld",
        "m30_l3",
        "cis_l3_la",
        "cis_l3_lb",
        "cis_l3_lc",
        "cis_l3_ld",
    )

    x_position = 0
    _x_names = []
    all_values = []
    for struct in to_plot:
        topo, l1, l2 = name_parser(struct)

        s_values = results_dict[struct]
        if len(s_values) == 0:
            continue
        g_sasa = s_values[yproperty]
        g_sasa = conv(g_sasa)
        x_position += 1
        c = cms[topo][0]
        if l2 is None:
            name = f"{topo} {l1}"
        else:
            name = f"{topo} {l1} {l2}"
        if "cis" in name:
            num_metals = 2
        else:
            num_metals = int(topo.split("m")[-1])
        # Into kJ/mol-1 per kelvin
        # y = -(g_sasa * 2625.5) / 298.15
        # y = y / num_metals
        # Nah, just use G_sasa in kJmol-1 (assuming iit is ~-TdeltaS)
        y = (g_sasa * 2625.5) / num_metals

        _x_names.append((x_position, name))
        # input()
        all_values.append((x_position, y, c))
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


def plot_all_contributions(results_dict, outname):
    l1s = {"l1": "m6_l1", "l2": "m12_l2"}
    l2s = {"la": "m2_la", "lb": "m2_lb", "lc": "m2_lc", "ld": "m2_ld"}
    to_plot = {
        "cis_l1_la": {"het_stoich": 6, "l1_stoich": 1, "l2_stoich": 3},
        "cis_l1_lb": {"het_stoich": 6, "l1_stoich": 1, "l2_stoich": 3},
        "cis_l1_lc": {"het_stoich": 6, "l1_stoich": 1, "l2_stoich": 3},
        "cis_l1_ld": {"het_stoich": 6, "l1_stoich": 1, "l2_stoich": 3},
        "cis_l2_la": {"het_stoich": 12, "l1_stoich": 1, "l2_stoich": 6},
        "cis_l2_lb": {"het_stoich": 12, "l1_stoich": 1, "l2_stoich": 6},
        "cis_l2_lc": {"het_stoich": 12, "l1_stoich": 1, "l2_stoich": 6},
        "cis_l2_ld": {"het_stoich": 12, "l1_stoich": 1, "l2_stoich": 6},
    }
    symbol_convert = {
        "xtb_solv_opt_dmsoenergy_au": "E",
        "xtb_solv_opt_dmsofreeenergy_au": "G",
        "xtb_solv_opt_dmsoenthalpy_au": "H",
        "xtb_gsolv_au": "G_solv",
        "xtb_gsasa_au": "G_SASA",
        "xtb_ssasa": "TS_SASA",
    }
    symbols = {
        "xtb_solv_opt_dmsoenergy_au": [],
        "xtb_solv_opt_dmsofreeenergy_au": [],
        "xtb_solv_opt_dmsoenthalpy_au": [],
        "xtb_gsolv_au": [],
        "xtb_gsasa_au": [],
    }

    for het_structure in to_plot:
        het_dict = results_dict[het_structure]
        topo, l1, l2 = name_parser(het_structure)
        l1_dict = results_dict[l1s[l1]]
        l2_dict = results_dict[l2s[l2]]
        for symbol in symbols:
            het_value = het_dict[symbol]
            l1_value = l1_dict[symbol]
            l2_value = l2_dict[symbol]
            het_stoich = to_plot[het_structure]["het_stoich"]
            l1_stoich = to_plot[het_structure]["l1_stoich"]
            l2_stoich = to_plot[het_structure]["l2_stoich"]
            heteroleptics = het_value * het_stoich
            homoleptics = (l1_value * l1_stoich) + (l2_value * l2_stoich)
            specific_exchange_energy = homoleptics - heteroleptics
            specific_exchange_energy = specific_exchange_energy / het_stoich
            specific_exchange_energy = specific_exchange_energy * 2625.5
            symbols[symbol].append(specific_exchange_energy)

    symbols["xtb_ssasa"] = [-i for i in symbols["xtb_gsasa_au"]]
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(16, 5))
    for symbol in symbols:
        # xdata = [i - symbols[symbol][0] for i in symbols[symbol]]
        xdata = symbols[symbol]
        if symbol == "xtb_gsolv_au":
            ax = axs[1]
        else:
            ax = axs[0]
        ax.plot(
            xdata,
            marker="o",
            markersize=12,
            markeredgecolor="k",
            # markerfacecolor="tab:blue",
            lw=2,
            ls="--",
            # c="k",
            label=symbol_convert[symbol],
        )

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("structure", fontsize=16)
    axs[0].set_ylabel("Eexch / het stoich [kj/mol-1]", fontsize=16)
    axs[0].legend(fontsize=16)
    axs[0].axhline(y=0, ls="--", alpha=0.4, c="k")

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("structure", fontsize=16)
    axs[1].set_ylabel("G_solv,exch / het stoich [kj/mol-1]", fontsize=16)
    axs[1].axhline(y=0, ls="--", alpha=0.4, c="k")

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_stab_energy(results_dict, outname, solvent=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    to_plot = (
        "m2_la",
        "m2_lb",
        "m2_lc",
        "m2_ld",
        "m6_l1",
        "cis_l1_la",
        "cis_l1_lb",
        "cis_l1_lc",
        "cis_l1_ld",
        # "m12_l2",
        # "cis_l2_la",
        # "cis_l2_lb",
        # "cis_l2_lc",
        # "cis_l2_ld",
        # "m30_l3",
        # "cis_l3_la",
        # "cis_l3_lb",
        # "cis_l3_lc",
        # "cis_l3_ld",
    )
    xs = []
    stabs = []
    e1dashes = []
    x_position = 0
    for system in to_plot:
        sdict = results_dict[system]
        topology, l1, l2 = name_parser(system)

        stab_energy = (sdict["stabilisation_energy"]) * 2625.5
        stabs.append(stab_energy)
        e1dashes.append(sdict["E1'"])
        xs.append((x_position, system))
        x_position += 1

    ax.plot(
        [i[0] for i in xs],
        stabs,
        markersize=8,
        marker="o",
        lw=2,
        label="E-E`",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("energy per metal [kJmol$^{-1}$]", fontsize=16)

    ax.set_xticks([i[0] for i in xs])
    ax.set_xticklabels([i[1] for i in xs], rotation=90)

    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.png"),
        dpi=360,
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
