"""Module for plotting functions."""

import logging
import pathlib
from collections import abc

import matplotlib.pyplot as plt
import numpy as np


def name_parser(name: str) -> tuple[str, str, str]:
    """Parse name."""
    splits = name.split("_")
    if len(splits) == 2:  # noqa: PLR2004
        topo, ligand1 = splits
        ligand2 = None
    elif len(splits) == 3:  # noqa: PLR2004
        topo, ligand1, ligand2 = splits

    return topo, ligand1, ligand2


def name_conversion() -> dict[str, str]:
    """Convert names."""
    return {
        "l1": r"1$^{\mathrm{DBF}}$",
        "l2": r"1$^{\mathrm{Ph}}$",
        "l3": r"1$^{\mathrm{Th}}$",
        "la": r"2$^{\mathrm{DBF}}$",
        "lb": r"2$^{\mathrm{Py}}$",
        "lc": r"2$^{\mathrm{Ph}}$",
        "ld": r"2$^{\mathrm{Th}}$",
        "e11": "e11",
        "e12": "e12",
        "e16": "e16",
        "e13": "e13",
        "e18": "e18",
        "e10": "e10",
        "e14": "e14",
        "e17": "e17",
    }


def expt_name_conversion(l1: str, l2: str) -> str:
    """Convert names."""
    return {
        ("e16", "e10"): "1",
        ("e16", "e17"): "2",
        ("e10", "e17"): "3",
        ("e11", "e10"): "4",
        ("e16", "e14"): "5",
        ("e18", "e14"): "6",
        ("e18", "e10"): "7",
        ("e12", "e10"): "8",
        ("e11", "e14"): "9",
        ("e12", "e14"): "10",
        ("e11", "e13"): "11",
        ("e12", "e13"): "12",
        ("e13", "e14"): "13",
        ("e11", "e12"): "14",
    }[(l1, l2)]


def c_and_m_properties() -> dict[str, tuple[str, str]]:
    """Get plot properties."""
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


def axes_labels(prop: str) -> tuple[str, tuple[float, float], abc.Callable]:
    """Axis label definitions for properties."""

    def no_conv(y: float | dict) -> float:
        return y

    def avg_strain_to_kjmol(y: dict) -> float:
        """Modify y."""
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
    results_dict: dict,
    dihedral_cutoff: float,
    output_path: pathlib.Path,
) -> None:
    """Plot ligand geom scores."""
    name = output_path.name
    logging.info("plotting: plot_all_geom_scores_simplified of %s", name)

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

        if meas == "g":
            ec = "k"
            abottom = np.zeros(len(pair_to_x))

        else:
            ec = "w"
            abottom = bottom

        p = ax.bar(
            x,
            y,
            width,
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

        bottom += np.asarray(y)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("deviation", fontsize=16)
    ax.set_ylim(0, 0.9)

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
    fig.savefig(output_path, dpi=720, bbox_inches="tight")
    plt.close()


def gs_table(results_dict: dict, dihedral_cutoff: float) -> None:
    """Print g table."""
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

            if geom_score < 0.45:  # noqa: PLR2004
                good_geoms += 1

        logging.info(
            "%s: %s, %s %s (%s) ",
            pair_name,
            round(min_geom_score, 2),
            round((good_geoms / total_tested) * 100, 0),
            round(np.mean(geom_scores), 2),
            round(np.std(geom_scores), 2),
        )


def simple_beeswarm(
    y: np.ndarray,
    nbins: int | None = None,
    width: float = 1.0,
) -> np.ndarray:
    """Returns coordinates for the points in y in a bee swarm plot."""
    y = np.asarray(y)
    if nbins is None:
        # nbins = len(y) // 6  # noqa: ERA001
        nbins = np.ceil(len(y) / 6).astype(int)

    # Get upper bounds of bins
    x = np.zeros(len(y))

    nn, ybins = np.histogram(y, bins=nbins)
    nmax = nn.max()

    # Divide indices into bins
    ibs = []  # np.nonzero((y>=ybins[0])*(y<=ybins[1]))[0]]
    for ymin, ymax in zip(ybins[:-1], ybins[1:], strict=False):  # noqa: RUF007
        i = np.nonzero((y > ymin) * (y <= ymax))[0]
        ibs.append(i)

    # Assign x indices
    dx = width / (nmax // 2)
    for i in ibs:
        yy = y[i]
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(yy)]  # noqa: PLW2901
            a = i[j::2]
            b = i[j + 1 :: 2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def plot_all_ligand_pairings_simplified(
    results_dict: dict,
    dihedral_cutoff: float,
    output_path: pathlib.Path,
) -> None:
    """Plot ligand pairings."""
    logging.info("plotting: plot_all_ligand_pairings ")

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

        if meas == "g":
            ec = "k"
            abottom = np.zeros(12)

        else:
            ec = "w"
            abottom = bottom

        p = ax.bar(
            x,
            y,
            width,
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

        bottom += np.asarray(y)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("deviation", fontsize=16)
    ax.set_ylim(0, 1.6)
    ax.axvline(x=3.5, c="gray", lw=2, linestyle="--")
    ax.axvline(x=7.5, c="gray", lw=2, linestyle="--")
    ax.text(x=1.2, y=1.5, s=name_conversion()["l1"], fontsize=16)
    ax.text(x=5.2, y=1.5, s=name_conversion()["l2"], fontsize=16)
    ax.text(x=9.2, y=1.5, s=name_conversion()["l3"], fontsize=16)

    ax.set_xticks(range(12))
    ax.set_xticklabels(
        [f"{name_conversion()[i[1]]}" for i in pair_to_x],
    )

    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(output_path, dpi=720, bbox_inches="tight")
    plt.close()


def plot_all_ligand_pairings(
    results_dict: dict,
    dihedral_cutoff: float,
    output_path: pathlib.Path,
) -> None:
    """Plot ligand pairings."""
    logging.info("plotting: plot_all_ligand_pairings ")

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
        ax.set_ylabel("deviation", fontsize=16)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1.6)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["|$a-1$|", "|$l-1$|", "$g$"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=720, bbox_inches="tight")
    plt.close()


def plot_all_ligand_pairings_2dhist(
    results_dict: dict,
    dihedral_cutoff: float,
    output_path: pathlib.Path,
) -> None:
    """Plot ligand pairings."""
    logging.info("plotting: plot_all_ligand_pairings")

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
    fig.savefig(output_path, dpi=720, bbox_inches="tight")
    plt.close()


def plot_all_ligand_pairings_conformers(
    results_dict: dict,
    structure_results: dict,
    dihedral_cutoff: float,
    output_path: pathlib.Path,
) -> None:
    """Plot ligand pairings."""
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

        for cid in l_confs:
            ax.plot(
                (l_confs[cid]["N1"][0], l_confs[cid]["N2"][0]),
                (l_confs[cid]["N1"][1], l_confs[cid]["N2"][1]),
                c="#374A67",
                alpha=0.2,
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
                lw=3,
            )
            ax.plot(
                (l_confs[cid]["N2"][0], l_confs[cid]["B2"][0]),
                (l_confs[cid]["N2"][1], l_confs[cid]["B2"][1]),
                c="#374A67",
                alpha=0.2,
                lw=3,
            )

        for cid in s_confs:
            ax.plot(
                (s_confs[cid]["N1"][0], s_confs[cid]["N2"][0]),
                (s_confs[cid]["N1"][1], s_confs[cid]["N2"][1]),
                c="#F0803C",
                alpha=0.2,
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
                lw=3,
            )
            ax.plot(
                (s_confs[cid]["N2"][0], s_confs[cid]["B2"][0]),
                (s_confs[cid]["N2"][1], s_confs[cid]["B2"][1]),
                c="#F0803C",
                alpha=0.2,
                lw=3,
            )

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
    fig.savefig(output_path, dpi=720, bbox_inches="tight")
    plt.close()


def plot_all_ligand_pairings_2dhist_fig5(
    results_dict: dict,
    dihedral_cutoff: float,
    output_path: pathlib.Path,
) -> None:
    """Plot ligand pairings."""
    logging.info("plotting: plot_all_ligand_pairings ")

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
    fig.savefig(output_path, dpi=720, bbox_inches="tight")
    plt.close()


def plot_single_distribution(
    results_dict: dict,
    output_path: pathlib.Path,
    yproperty: str,
) -> None:
    """Plot ligand pairings."""
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
    fig.savefig(output_path, dpi=720, bbox_inches="tight")
    plt.close()
