#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for new stk topologies.

Author: Andrew Tarzia

"""


import stk


def ligand_cage_topologies():
    return {
        "l1": ("m6",),
        "l2": ("m12",),
        "l3": ("m24", "m30"),
        "la": ("m2", "m3", "m4"),
        "lb": ("m2", "m3", "m4"),
        "lc": ("m2", "m3", "m4"),
        "ld": ("m2", "m3", "m4"),
    }


class M30L60(stk.cage.Cage):

    _vertex_prototypes = ()
    _edge_prototypes = ()
