#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for plotting functions.

Author: Andrew Tarzia

"""

import matplotlib.pyplot as plt

# import matplotlib.colors as colors
import matplotlib as mpl
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


def plot_geom_scores(
    results_dict,
    outname,
):

    fig, ax = plt.subplots(figsize=(8, 5))

    min_small_e = min(
        [results_dict[i]["small_energy"] for i in results_dict]
    )
    min_large_e = min(
        [results_dict[i]["large_energy"] for i in results_dict]
    )
    print(min_large_e, min_small_e)

    all_xs = []
    all_ys = []
    for cids, cidl in results_dict:
        rdict = results_dict[(cids, cidl)]
        small_strain = (rdict["small_energy"] - min_small_e) * 2625.5
        large_strain = (rdict["large_energy"] - min_large_e) * 2625.5
        if rdict["geom_score"] > 0:
            all_xs.append(rdict["geom_score"])
            all_ys.append((small_strain + large_strain))

    ax.axvline(x=2, c="gray", linestyle="--")
    ax.axhline(y=5, lw=2, c="gray", linestyle="--")

    ax.scatter(
        [i for i in all_xs],
        [i for i in all_ys],
        c="teal",
        s=30,
        alpha=1.0,
        edgecolors="none",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("geom score", fontsize=16)
    ax.set_ylabel("strain [kJmol-1]", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_ligand_pairing(
    results_dict,
    outname,
):

    fig, ax = plt.subplots(figsize=(8, 5))

    min_small_e = min(
        [results_dict[i]["small_energy"] for i in results_dict]
    )
    min_large_e = min(
        [results_dict[i]["large_energy"] for i in results_dict]
    )
    print(min_large_e, min_small_e)

    all_xs = []
    all_ys = []
    all_cs = []
    num_points_total = 0
    num_good = 0
    for cids, cidl in results_dict:
        rdict = results_dict[(cids, cidl)]
        small_strain = (rdict["small_energy"] - min_small_e) * 2625.5
        large_strain = (rdict["large_energy"] - min_large_e) * 2625.5
        all_xs.append(rdict["angle_ratio"])
        all_ys.append(rdict["length_ratio"])
        all_cs.append((small_strain + large_strain))
        num_points_total += 1
        if abs(rdict["geom_score"]) < 2:
            num_good += 1

    ax.scatter(
        all_xs,
        all_ys,
        c=all_cs,
        s=20,
        edgecolor="none",
        cmap="Blues_r",
        vmin=0,
        vmax=5,
    )
    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cmap = mpl.cm.Blues_r
    norm = mpl.colors.Normalize(vmin=0, vmax=5)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("strain [kJmol-1]", fontsize=16)

    ax.set_title(f"{num_good} of {num_points_total}", fontsize=16)
    ax.axhline(y=1, c="gray", lw=1, linestyle="--")
    ax.axvline(x=1, c="gray", lw=1, linestyle="--")

    # xlim = (0, 2)
    # ylim = (0, 2)
    # norm = colors.LogNorm()
    # cs = [(1.0, 1.0, 1.0), (255 / 255, 87 / 255, 51 / 255)]
    # cmap = colors.LinearSegmentedColormap.from_list("test", cs, N=10)
    # hist = ax.hist2d(
    #     all_xs,
    #     all_ys,
    #     bins=[40, 40],
    #     range=[xlim, ylim],
    #     density=False,
    #     norm=norm,
    #     cmap=cmap,
    # )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("angle ratio []", fontsize=16)
    ax.set_ylabel("length ratio []", fontsize=16)
    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 2)
    # cbar = fig.colorbar(hist[3], ax=ax)
    # cbar.ax.set_ylabel("count", fontsize=16)
    # cbar.ax.tick_params(labelsize=16)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
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
