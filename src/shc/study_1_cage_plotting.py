"""Module for plotting functions."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from study_1_plotting import axes_labels


def plot_strain_pore_sasa(results_dict, outname):
    raise NotImplementedError
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
    raise NotImplementedError
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
    raise NotImplementedError
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
    raise NotImplementedError
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
    raise NotImplementedError
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
    raise NotImplementedError
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
    raise NotImplementedError
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
    raise NotImplementedError
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
    raise NotImplementedError
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
    raise NotImplementedError
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
    raise NotImplementedError
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
