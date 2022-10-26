#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for plotting functions.

Author: Andrew Tarzia

"""

import matplotlib.pyplot as plt
import os

from env_set import figu_path
from utilities import name_parser


def c_and_m_properties():
    return {
        "m2": ("#212738", "o"),
        "m3": ("#84E296", "o"),
        "m4": ("#387780", "o"),
        "cis": ("#F97068", "o"),
        "trans": ("#57C4E5", "o"),
    }


def axes_labels(prop):
    def no_conv(y):
        return y

    def sum_strain(y):
        return sum(y.values()) * 2625.5

    return {
        "min_order_param": ("min op.", (0, 1), no_conv),
        "pore_diameter_opt": (
            r"pywindow pore volume [$\mathrm{\AA}^{3}$]",
            (0, 10),
            no_conv,
        ),
        "xtb_lig_strain": (
            "sum strain energy [kJ mol-1]",
            (0, 300),
            sum_strain,
        ),
    }[prop]


def plot_property(results_dict, outname, yproperty, ignore_topos=None):

    if ignore_topos is None:
        ignore_topos = ()

    cms = c_and_m_properties()
    lab_prop = axes_labels(yproperty)
    conv = lab_prop[2]

    sorted_names = sorted(results_dict.keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    x_position = 0
    _x_names = []
    all_values = []
    for struct in sorted_names:
        topo, l1, l2 = name_parser(struct)
        if topo in ignore_topos:
            continue
        s_values = results_dict[struct]
        try:
            y = s_values[yproperty]
        except KeyError:
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

    ax.scatter(
        x=[i[0] for i in all_values],
        y=[i[1] for i in all_values],
        c=[i[2] for i in all_values],
        edgecolors="k",
        s=180,
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

    ignore_topos = ("m2", "m3", "m4")

    lab_prop = axes_labels(yproperty)
    conv = lab_prop[2]

    sorted_names = sorted(results_dict.keys())

    fig, ax = plt.subplots(figsize=(5, 5))

    _x_names = []
    for struct in sorted_names:
        topo, l1, l2 = name_parser(struct)
        name = f"{l1} {l2}"
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

        # text.
        _x_names.append(name)
        ax.scatter(
            x=trans,
            y=cis,
            c="gold",
            edgecolors="k",
            s=180,
        )
        if trans > lab_prop[1][1] or cis > lab_prop[1][1]:
            continue
        ax.text(trans, cis, name, fontsize=16)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("trans", fontsize=16)
    ax.set_ylabel("cis", fontsize=16)
    ax.set_title(lab_prop[0], fontsize=16)

    ax.set_xlim(lab_prop[1])
    ax.set_ylim(lab_prop[1])

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_exchange_reactions(rxns, hs, outname):

    fig, ax = plt.subplots(figsize=(8, 5))
    x_values = []
    for i, rxn in enumerate(rxns):
        print(rxn)
        # l1_prefix = int(rxn['l1_prefix'][1])
        # l2_prefix = int(rxn['l2_prefix'][1])
        # r_string = (
        #     f"{rxn['lhs_stoich']}[Pd2 L2 L'2] "
        #     "<-> "
        #     f"{rxn['l1_stoich']}[Pd{l1_prefix} "
        #     f"L{l1_prefix*2}] + "
        #     f"{rxn['l2_stoich']}[Pd{l2_prefix} "
        #     f"L'{l2_prefix*2}]"
        # )
        r_string = (
            f"{rxn['lhs_stoich']} "
            "<-> \n"
            f"{rxn['l1_stoich']}{rxn['l1_prefix']} + "
            f"{rxn['l2_stoich']}{rxn['l2_prefix']} "
        )
        print(r_string)
        r_energy = float(rxn["lhs"]) - float(rxn["rhs"])
        r_energy = r_energy / int(rxn["lhs_stoich"])
        r_energy = r_energy * 2625.5
        ax.scatter(
            x=i,
            y=r_energy,
            c="gold",
            edgecolors="k",
            s=180,
        )
        # ax.text(i-2, r_energy, r_string, fontsize=16)
        x_values.append((i, r_string))

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_title(f"{rxn['l1']} + {rxn['l2']}", fontsize=16)
    ax.set_xlabel("reaction", fontsize=16)
    ax.set_ylabel(
        "energy per heteroleptic cage [kJ mol-1]",
        fontsize=16,
    )
    ax.axhline(y=0, lw=2, c="k")

    ax.set_xticks([i[0] for i in x_values])
    ax.set_xticklabels([i[1] for i in x_values], rotation=45)

    # ax.set_xlim(lab_prop[1])
    # ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figu_path(), f"{outname}.pdf"),
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
