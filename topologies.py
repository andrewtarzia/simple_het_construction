#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for new stk topologies.

Author: Andrew Tarzia

"""


import stk


def heteroleptic_cages():
    return (
        ("l1", "la"),
        ("l1", "lb"),
        ("l1", "lc"),
        ("l1", "ld"),
        ("l2", "la"),
        ("l2", "lb"),
        ("l2", "lc"),
        ("l2", "ld"),
        ("l3", "la"),
        ("l3", "lb"),
        ("l3", "lc"),
        ("l3", "ld"),
    )


def ligand_cage_topologies():
    return {
        "l1": ("m2", "m3", "m4", "m6"),
        "l2": ("m12",),
        "l3": ("m24", "m30"),
        "la": ("m2",),
        "lb": ("m2",),
        "lc": ("m2",),
        "ld": ("m2",),
    }


def erxn_cage_topologies():
    return {
        "l1": "m6",
        "l2": "m12",
        "l3": "m24",
        "la": "m2",
        "lb": "m2",
        "lc": "m2",
        "ld": "m2",
    }


class M30L60(stk.cage.Cage):
    def _get_scale(self, building_block_vertices):
        return 0.7

    _vertex_prototypes = (
        stk.cage.LinearVertex(
            id=0,
            position=[9.9, -9.3, -34.3],
        ),
        stk.cage.LinearVertex(
            id=1,
            position=[-3.6, -8.7, -35.2],
        ),
        stk.cage.LinearVertex(
            id=2,
            position=[25.1, -21.3, 16.9],
        ),
        stk.cage.LinearVertex(
            id=3,
            position=[-25.1, -21.3, -16.9],
        ),
        stk.cage.LinearVertex(
            id=4,
            position=[34.1, -6.8, 12.2],
        ),
        stk.cage.LinearVertex(
            id=5,
            position=[-34.1, -6.8, -12.2],
        ),
        stk.cage.LinearVertex(
            id=6,
            position=[34.9, -6.8, -6.1],
        ),
        stk.cage.LinearVertex(
            id=7,
            position=[-34.9, -6.8, 6.1],
        ),
        stk.cage.LinearVertex(
            id=8,
            position=[27.6, -22.1, -13.3],
        ),
        stk.cage.LinearVertex(
            id=9,
            position=[-27.6, -22.1, 13.3],
        ),
        stk.cage.LinearVertex(
            id=10,
            position=[20.4, -30.2, 1.5],
        ),
        stk.cage.LinearVertex(
            id=11,
            position=[-20.4, -30.2, -1.5],
        ),
        stk.cage.LinearVertex(
            id=12,
            position=[9.8, -33.6, -5.9],
        ),
        stk.cage.LinearVertex(
            id=13,
            position=[-9.8, -33.6, 5.9],
        ),
        stk.cage.LinearVertex(
            id=14,
            position=[8.7, -33.4, 7.3],
        ),
        stk.cage.LinearVertex(
            id=15,
            position=[-8.7, -33.4, -7.3],
        ),
        stk.cage.LinearVertex(
            id=16,
            position=[16.8, -25.6, -20.2],
        ),
        stk.cage.LinearVertex(
            id=17,
            position=[-16.8, -25.6, 20.2],
        ),
        stk.cage.LinearVertex(
            id=18,
            position=[23.6, -14.9, -23.8],
        ),
        stk.cage.LinearVertex(
            id=19,
            position=[-23.6, -14.9, 23.8],
        ),
        stk.cage.LinearVertex(
            id=20,
            position=[2.5, -19.5, -30.1],
        ),
        stk.cage.LinearVertex(
            id=21,
            position=[-2.5, -19.5, 30.1],
        ),
        stk.cage.LinearVertex(
            id=22,
            position=[-13.3, -24.8, -22.4],
        ),
        stk.cage.LinearVertex(
            id=23,
            position=[13.3, -24.8, 22.4],
        ),
        stk.cage.LinearVertex(
            id=24,
            position=[-19.6, -14.3, -27.4],
        ),
        stk.cage.LinearVertex(
            id=25,
            position=[19.6, -14.3, 27.4],
        ),
        stk.cage.NonLinearVertex(
            id=26,
            position=[30.9, -11.5, -15.7],
        ),
        stk.cage.NonLinearVertex(
            id=27,
            position=[-30.9, -11.5, 15.7],
        ),
        stk.cage.NonLinearVertex(
            id=28,
            position=[28.5, -11.2, 21.1],
        ),
        stk.cage.NonLinearVertex(
            id=29,
            position=[-28.5, -11.2, -21.1],
        ),
        stk.cage.NonLinearVertex(
            id=30,
            position=[17.7, -28.7, 12.6],
        ),
        stk.cage.NonLinearVertex(
            id=31,
            position=[-17.7, -28.7, -12.6],
        ),
        stk.cage.NonLinearVertex(
            id=32,
            position=[19.7, -29.5, -9.9],
        ),
        stk.cage.NonLinearVertex(
            id=33,
            position=[-19.7, -29.5, 9.9],
        ),
        stk.cage.NonLinearVertex(
            id=34,
            position=[13.7, -18.3, -28.4],
        ),
        stk.cage.NonLinearVertex(
            id=35,
            position=[-13.7, -18.3, 28.4],
        ),
        stk.cage.NonLinearVertex(
            id=36,
            position=[-8.8, -17.4, -29.9],
        ),
        stk.cage.NonLinearVertex(
            id=37,
            position=[8.8, -17.4, 29.9],
        ),
        stk.cage.NonLinearVertex(
            id=38,
            position=[0.0, -34.9, 0.0],
        ),
        stk.cage.LinearVertex(
            id=39,
            position=[-9.9, -9.3, 34.3],
        ),
        stk.cage.LinearVertex(
            id=40,
            position=[3.6, -8.7, 35.2],
        ),
        stk.cage.LinearVertex(
            id=41,
            position=[9.9, 9.3, -34.3],
        ),
        stk.cage.LinearVertex(
            id=42,
            position=[-3.6, 8.7, -35.2],
        ),
        stk.cage.NonLinearVertex(
            id=43,
            position=[3.6, 0.0, -36.6],
        ),
        stk.cage.LinearVertex(
            id=44,
            position=[-25.1, 21.3, -16.9],
        ),
        stk.cage.LinearVertex(
            id=45,
            position=[25.1, 21.3, 16.9],
        ),
        stk.cage.LinearVertex(
            id=46,
            position=[-34.1, 6.8, -12.2],
        ),
        stk.cage.LinearVertex(
            id=47,
            position=[34.1, 6.8, 12.2],
        ),
        stk.cage.LinearVertex(
            id=48,
            position=[-34.9, 6.8, 6.1],
        ),
        stk.cage.LinearVertex(
            id=49,
            position=[34.9, 6.8, -6.1],
        ),
        stk.cage.LinearVertex(
            id=50,
            position=[-27.6, 22.1, 13.3],
        ),
        stk.cage.LinearVertex(
            id=51,
            position=[27.6, 22.1, -13.3],
        ),
        stk.cage.LinearVertex(
            id=52,
            position=[-20.4, 30.2, -1.5],
        ),
        stk.cage.LinearVertex(
            id=53,
            position=[20.4, 30.2, 1.5],
        ),
        stk.cage.LinearVertex(
            id=54,
            position=[-9.8, 33.6, 5.9],
        ),
        stk.cage.LinearVertex(
            id=55,
            position=[9.8, 33.6, -5.9],
        ),
        stk.cage.LinearVertex(
            id=56,
            position=[-8.7, 33.4, -7.3],
        ),
        stk.cage.LinearVertex(
            id=57,
            position=[8.7, 33.4, 7.3],
        ),
        stk.cage.LinearVertex(
            id=58,
            position=[-16.8, 25.6, 20.2],
        ),
        stk.cage.LinearVertex(
            id=59,
            position=[16.8, 25.6, -20.2],
        ),
        stk.cage.LinearVertex(
            id=60,
            position=[-23.6, 14.9, 23.8],
        ),
        stk.cage.LinearVertex(
            id=61,
            position=[23.6, 14.9, -23.8],
        ),
        stk.cage.LinearVertex(
            id=62,
            position=[-2.5, 19.5, 30.1],
        ),
        stk.cage.LinearVertex(
            id=63,
            position=[2.5, 19.5, -30.1],
        ),
        stk.cage.LinearVertex(
            id=64,
            position=[13.3, 24.8, 22.4],
        ),
        stk.cage.LinearVertex(
            id=65,
            position=[-13.3, 24.8, -22.4],
        ),
        stk.cage.LinearVertex(
            id=66,
            position=[19.6, 14.3, 27.4],
        ),
        stk.cage.LinearVertex(
            id=67,
            position=[-19.6, 14.3, -27.4],
        ),
        stk.cage.LinearVertex(
            id=68,
            position=[-29.6, 0.0, -22.9],
        ),
        stk.cage.LinearVertex(
            id=69,
            position=[29.6, 0.0, 22.9],
        ),
        stk.cage.LinearVertex(
            id=70,
            position=[-31.3, 0.0, 16.9],
        ),
        stk.cage.LinearVertex(
            id=71,
            position=[31.3, 0.0, -16.9],
        ),
        stk.cage.NonLinearVertex(
            id=72,
            position=[-30.9, 11.5, 15.7],
        ),
        stk.cage.NonLinearVertex(
            id=73,
            position=[30.9, 11.5, -15.7],
        ),
        stk.cage.NonLinearVertex(
            id=74,
            position=[-28.5, 11.2, -21.1],
        ),
        stk.cage.NonLinearVertex(
            id=75,
            position=[28.5, 11.2, 21.1],
        ),
        stk.cage.NonLinearVertex(
            id=76,
            position=[-35.9, 0.0, -3.2],
        ),
        stk.cage.NonLinearVertex(
            id=77,
            position=[35.9, 0.0, 3.2],
        ),
        stk.cage.NonLinearVertex(
            id=78,
            position=[-17.7, 28.7, -12.6],
        ),
        stk.cage.NonLinearVertex(
            id=79,
            position=[17.7, 28.7, 12.6],
        ),
        stk.cage.NonLinearVertex(
            id=80,
            position=[-19.7, 29.5, 9.9],
        ),
        stk.cage.NonLinearVertex(
            id=81,
            position=[19.7, 29.5, -9.9],
        ),
        stk.cage.NonLinearVertex(
            id=82,
            position=[-13.7, 18.3, 28.4],
        ),
        stk.cage.NonLinearVertex(
            id=83,
            position=[13.7, 18.3, -28.4],
        ),
        stk.cage.NonLinearVertex(
            id=84,
            position=[8.8, 17.4, 29.9],
        ),
        stk.cage.NonLinearVertex(
            id=85,
            position=[-8.8, 17.4, -29.9],
        ),
        stk.cage.NonLinearVertex(
            id=86,
            position=[0.0, 34.9, 0.0],
        ),
        stk.cage.LinearVertex(
            id=87,
            position=[-9.9, 9.3, 34.3],
        ),
        stk.cage.LinearVertex(
            id=88,
            position=[3.6, 8.7, 35.2],
        ),
        stk.cage.NonLinearVertex(
            id=89,
            position=[-3.6, 0.0, 36.6],
        ),
    )
    _edge_prototypes = (
        stk.Edge(
            id=0,
            vertex1=_vertex_prototypes[6],
            vertex2=_vertex_prototypes[26],
        ),
        stk.Edge(
            id=1,
            vertex1=_vertex_prototypes[8],
            vertex2=_vertex_prototypes[26],
        ),
        stk.Edge(
            id=2,
            vertex1=_vertex_prototypes[18],
            vertex2=_vertex_prototypes[26],
        ),
        stk.Edge(
            id=3,
            vertex1=_vertex_prototypes[26],
            vertex2=_vertex_prototypes[71],
        ),
        stk.Edge(
            id=4,
            vertex1=_vertex_prototypes[7],
            vertex2=_vertex_prototypes[27],
        ),
        stk.Edge(
            id=5,
            vertex1=_vertex_prototypes[9],
            vertex2=_vertex_prototypes[27],
        ),
        stk.Edge(
            id=6,
            vertex1=_vertex_prototypes[19],
            vertex2=_vertex_prototypes[27],
        ),
        stk.Edge(
            id=7,
            vertex1=_vertex_prototypes[27],
            vertex2=_vertex_prototypes[70],
        ),
        stk.Edge(
            id=8,
            vertex1=_vertex_prototypes[2],
            vertex2=_vertex_prototypes[28],
        ),
        stk.Edge(
            id=9,
            vertex1=_vertex_prototypes[4],
            vertex2=_vertex_prototypes[28],
        ),
        stk.Edge(
            id=10,
            vertex1=_vertex_prototypes[25],
            vertex2=_vertex_prototypes[28],
        ),
        stk.Edge(
            id=11,
            vertex1=_vertex_prototypes[28],
            vertex2=_vertex_prototypes[69],
        ),
        stk.Edge(
            id=12,
            vertex1=_vertex_prototypes[3],
            vertex2=_vertex_prototypes[29],
        ),
        stk.Edge(
            id=13,
            vertex1=_vertex_prototypes[5],
            vertex2=_vertex_prototypes[29],
        ),
        stk.Edge(
            id=14,
            vertex1=_vertex_prototypes[24],
            vertex2=_vertex_prototypes[29],
        ),
        stk.Edge(
            id=15,
            vertex1=_vertex_prototypes[29],
            vertex2=_vertex_prototypes[68],
        ),
        stk.Edge(
            id=16,
            vertex1=_vertex_prototypes[2],
            vertex2=_vertex_prototypes[30],
        ),
        stk.Edge(
            id=17,
            vertex1=_vertex_prototypes[10],
            vertex2=_vertex_prototypes[30],
        ),
        stk.Edge(
            id=18,
            vertex1=_vertex_prototypes[14],
            vertex2=_vertex_prototypes[30],
        ),
        stk.Edge(
            id=19,
            vertex1=_vertex_prototypes[23],
            vertex2=_vertex_prototypes[30],
        ),
        stk.Edge(
            id=20,
            vertex1=_vertex_prototypes[3],
            vertex2=_vertex_prototypes[31],
        ),
        stk.Edge(
            id=21,
            vertex1=_vertex_prototypes[11],
            vertex2=_vertex_prototypes[31],
        ),
        stk.Edge(
            id=22,
            vertex1=_vertex_prototypes[15],
            vertex2=_vertex_prototypes[31],
        ),
        stk.Edge(
            id=23,
            vertex1=_vertex_prototypes[22],
            vertex2=_vertex_prototypes[31],
        ),
        stk.Edge(
            id=24,
            vertex1=_vertex_prototypes[8],
            vertex2=_vertex_prototypes[32],
        ),
        stk.Edge(
            id=25,
            vertex1=_vertex_prototypes[10],
            vertex2=_vertex_prototypes[32],
        ),
        stk.Edge(
            id=26,
            vertex1=_vertex_prototypes[12],
            vertex2=_vertex_prototypes[32],
        ),
        stk.Edge(
            id=27,
            vertex1=_vertex_prototypes[16],
            vertex2=_vertex_prototypes[32],
        ),
        stk.Edge(
            id=28,
            vertex1=_vertex_prototypes[9],
            vertex2=_vertex_prototypes[33],
        ),
        stk.Edge(
            id=29,
            vertex1=_vertex_prototypes[11],
            vertex2=_vertex_prototypes[33],
        ),
        stk.Edge(
            id=30,
            vertex1=_vertex_prototypes[13],
            vertex2=_vertex_prototypes[33],
        ),
        stk.Edge(
            id=31,
            vertex1=_vertex_prototypes[17],
            vertex2=_vertex_prototypes[33],
        ),
        stk.Edge(
            id=32,
            vertex1=_vertex_prototypes[0],
            vertex2=_vertex_prototypes[34],
        ),
        stk.Edge(
            id=33,
            vertex1=_vertex_prototypes[16],
            vertex2=_vertex_prototypes[34],
        ),
        stk.Edge(
            id=34,
            vertex1=_vertex_prototypes[18],
            vertex2=_vertex_prototypes[34],
        ),
        stk.Edge(
            id=35,
            vertex1=_vertex_prototypes[20],
            vertex2=_vertex_prototypes[34],
        ),
        stk.Edge(
            id=36,
            vertex1=_vertex_prototypes[17],
            vertex2=_vertex_prototypes[35],
        ),
        stk.Edge(
            id=37,
            vertex1=_vertex_prototypes[19],
            vertex2=_vertex_prototypes[35],
        ),
        stk.Edge(
            id=38,
            vertex1=_vertex_prototypes[21],
            vertex2=_vertex_prototypes[35],
        ),
        stk.Edge(
            id=39,
            vertex1=_vertex_prototypes[35],
            vertex2=_vertex_prototypes[39],
        ),
        stk.Edge(
            id=40,
            vertex1=_vertex_prototypes[1],
            vertex2=_vertex_prototypes[36],
        ),
        stk.Edge(
            id=41,
            vertex1=_vertex_prototypes[20],
            vertex2=_vertex_prototypes[36],
        ),
        stk.Edge(
            id=42,
            vertex1=_vertex_prototypes[22],
            vertex2=_vertex_prototypes[36],
        ),
        stk.Edge(
            id=43,
            vertex1=_vertex_prototypes[24],
            vertex2=_vertex_prototypes[36],
        ),
        stk.Edge(
            id=44,
            vertex1=_vertex_prototypes[21],
            vertex2=_vertex_prototypes[37],
        ),
        stk.Edge(
            id=45,
            vertex1=_vertex_prototypes[23],
            vertex2=_vertex_prototypes[37],
        ),
        stk.Edge(
            id=46,
            vertex1=_vertex_prototypes[25],
            vertex2=_vertex_prototypes[37],
        ),
        stk.Edge(
            id=47,
            vertex1=_vertex_prototypes[37],
            vertex2=_vertex_prototypes[40],
        ),
        stk.Edge(
            id=48,
            vertex1=_vertex_prototypes[12],
            vertex2=_vertex_prototypes[38],
        ),
        stk.Edge(
            id=49,
            vertex1=_vertex_prototypes[13],
            vertex2=_vertex_prototypes[38],
        ),
        stk.Edge(
            id=50,
            vertex1=_vertex_prototypes[14],
            vertex2=_vertex_prototypes[38],
        ),
        stk.Edge(
            id=51,
            vertex1=_vertex_prototypes[15],
            vertex2=_vertex_prototypes[38],
        ),
        stk.Edge(
            id=52,
            vertex1=_vertex_prototypes[0],
            vertex2=_vertex_prototypes[43],
        ),
        stk.Edge(
            id=53,
            vertex1=_vertex_prototypes[1],
            vertex2=_vertex_prototypes[43],
        ),
        stk.Edge(
            id=54,
            vertex1=_vertex_prototypes[41],
            vertex2=_vertex_prototypes[43],
        ),
        stk.Edge(
            id=55,
            vertex1=_vertex_prototypes[42],
            vertex2=_vertex_prototypes[43],
        ),
        stk.Edge(
            id=56,
            vertex1=_vertex_prototypes[48],
            vertex2=_vertex_prototypes[72],
        ),
        stk.Edge(
            id=57,
            vertex1=_vertex_prototypes[50],
            vertex2=_vertex_prototypes[72],
        ),
        stk.Edge(
            id=58,
            vertex1=_vertex_prototypes[60],
            vertex2=_vertex_prototypes[72],
        ),
        stk.Edge(
            id=59,
            vertex1=_vertex_prototypes[70],
            vertex2=_vertex_prototypes[72],
        ),
        stk.Edge(
            id=60,
            vertex1=_vertex_prototypes[49],
            vertex2=_vertex_prototypes[73],
        ),
        stk.Edge(
            id=61,
            vertex1=_vertex_prototypes[51],
            vertex2=_vertex_prototypes[73],
        ),
        stk.Edge(
            id=62,
            vertex1=_vertex_prototypes[61],
            vertex2=_vertex_prototypes[73],
        ),
        stk.Edge(
            id=63,
            vertex1=_vertex_prototypes[71],
            vertex2=_vertex_prototypes[73],
        ),
        stk.Edge(
            id=64,
            vertex1=_vertex_prototypes[44],
            vertex2=_vertex_prototypes[74],
        ),
        stk.Edge(
            id=65,
            vertex1=_vertex_prototypes[46],
            vertex2=_vertex_prototypes[74],
        ),
        stk.Edge(
            id=66,
            vertex1=_vertex_prototypes[67],
            vertex2=_vertex_prototypes[74],
        ),
        stk.Edge(
            id=67,
            vertex1=_vertex_prototypes[68],
            vertex2=_vertex_prototypes[74],
        ),
        stk.Edge(
            id=68,
            vertex1=_vertex_prototypes[45],
            vertex2=_vertex_prototypes[75],
        ),
        stk.Edge(
            id=69,
            vertex1=_vertex_prototypes[47],
            vertex2=_vertex_prototypes[75],
        ),
        stk.Edge(
            id=70,
            vertex1=_vertex_prototypes[66],
            vertex2=_vertex_prototypes[75],
        ),
        stk.Edge(
            id=71,
            vertex1=_vertex_prototypes[69],
            vertex2=_vertex_prototypes[75],
        ),
        stk.Edge(
            id=72,
            vertex1=_vertex_prototypes[5],
            vertex2=_vertex_prototypes[76],
        ),
        stk.Edge(
            id=73,
            vertex1=_vertex_prototypes[7],
            vertex2=_vertex_prototypes[76],
        ),
        stk.Edge(
            id=74,
            vertex1=_vertex_prototypes[46],
            vertex2=_vertex_prototypes[76],
        ),
        stk.Edge(
            id=75,
            vertex1=_vertex_prototypes[48],
            vertex2=_vertex_prototypes[76],
        ),
        stk.Edge(
            id=76,
            vertex1=_vertex_prototypes[4],
            vertex2=_vertex_prototypes[77],
        ),
        stk.Edge(
            id=77,
            vertex1=_vertex_prototypes[6],
            vertex2=_vertex_prototypes[77],
        ),
        stk.Edge(
            id=78,
            vertex1=_vertex_prototypes[47],
            vertex2=_vertex_prototypes[77],
        ),
        stk.Edge(
            id=79,
            vertex1=_vertex_prototypes[49],
            vertex2=_vertex_prototypes[77],
        ),
        stk.Edge(
            id=80,
            vertex1=_vertex_prototypes[44],
            vertex2=_vertex_prototypes[78],
        ),
        stk.Edge(
            id=81,
            vertex1=_vertex_prototypes[52],
            vertex2=_vertex_prototypes[78],
        ),
        stk.Edge(
            id=82,
            vertex1=_vertex_prototypes[56],
            vertex2=_vertex_prototypes[78],
        ),
        stk.Edge(
            id=83,
            vertex1=_vertex_prototypes[65],
            vertex2=_vertex_prototypes[78],
        ),
        stk.Edge(
            id=84,
            vertex1=_vertex_prototypes[45],
            vertex2=_vertex_prototypes[79],
        ),
        stk.Edge(
            id=85,
            vertex1=_vertex_prototypes[53],
            vertex2=_vertex_prototypes[79],
        ),
        stk.Edge(
            id=86,
            vertex1=_vertex_prototypes[57],
            vertex2=_vertex_prototypes[79],
        ),
        stk.Edge(
            id=87,
            vertex1=_vertex_prototypes[64],
            vertex2=_vertex_prototypes[79],
        ),
        stk.Edge(
            id=88,
            vertex1=_vertex_prototypes[50],
            vertex2=_vertex_prototypes[80],
        ),
        stk.Edge(
            id=89,
            vertex1=_vertex_prototypes[52],
            vertex2=_vertex_prototypes[80],
        ),
        stk.Edge(
            id=90,
            vertex1=_vertex_prototypes[54],
            vertex2=_vertex_prototypes[80],
        ),
        stk.Edge(
            id=91,
            vertex1=_vertex_prototypes[58],
            vertex2=_vertex_prototypes[80],
        ),
        stk.Edge(
            id=92,
            vertex1=_vertex_prototypes[51],
            vertex2=_vertex_prototypes[81],
        ),
        stk.Edge(
            id=93,
            vertex1=_vertex_prototypes[53],
            vertex2=_vertex_prototypes[81],
        ),
        stk.Edge(
            id=94,
            vertex1=_vertex_prototypes[55],
            vertex2=_vertex_prototypes[81],
        ),
        stk.Edge(
            id=95,
            vertex1=_vertex_prototypes[59],
            vertex2=_vertex_prototypes[81],
        ),
        stk.Edge(
            id=96,
            vertex1=_vertex_prototypes[58],
            vertex2=_vertex_prototypes[82],
        ),
        stk.Edge(
            id=97,
            vertex1=_vertex_prototypes[60],
            vertex2=_vertex_prototypes[82],
        ),
        stk.Edge(
            id=98,
            vertex1=_vertex_prototypes[62],
            vertex2=_vertex_prototypes[82],
        ),
        stk.Edge(
            id=99,
            vertex1=_vertex_prototypes[82],
            vertex2=_vertex_prototypes[87],
        ),
        stk.Edge(
            id=100,
            vertex1=_vertex_prototypes[41],
            vertex2=_vertex_prototypes[83],
        ),
        stk.Edge(
            id=101,
            vertex1=_vertex_prototypes[59],
            vertex2=_vertex_prototypes[83],
        ),
        stk.Edge(
            id=102,
            vertex1=_vertex_prototypes[61],
            vertex2=_vertex_prototypes[83],
        ),
        stk.Edge(
            id=103,
            vertex1=_vertex_prototypes[63],
            vertex2=_vertex_prototypes[83],
        ),
        stk.Edge(
            id=104,
            vertex1=_vertex_prototypes[62],
            vertex2=_vertex_prototypes[84],
        ),
        stk.Edge(
            id=105,
            vertex1=_vertex_prototypes[64],
            vertex2=_vertex_prototypes[84],
        ),
        stk.Edge(
            id=106,
            vertex1=_vertex_prototypes[66],
            vertex2=_vertex_prototypes[84],
        ),
        stk.Edge(
            id=107,
            vertex1=_vertex_prototypes[84],
            vertex2=_vertex_prototypes[88],
        ),
        stk.Edge(
            id=108,
            vertex1=_vertex_prototypes[42],
            vertex2=_vertex_prototypes[85],
        ),
        stk.Edge(
            id=109,
            vertex1=_vertex_prototypes[63],
            vertex2=_vertex_prototypes[85],
        ),
        stk.Edge(
            id=110,
            vertex1=_vertex_prototypes[65],
            vertex2=_vertex_prototypes[85],
        ),
        stk.Edge(
            id=111,
            vertex1=_vertex_prototypes[67],
            vertex2=_vertex_prototypes[85],
        ),
        stk.Edge(
            id=112,
            vertex1=_vertex_prototypes[54],
            vertex2=_vertex_prototypes[86],
        ),
        stk.Edge(
            id=113,
            vertex1=_vertex_prototypes[55],
            vertex2=_vertex_prototypes[86],
        ),
        stk.Edge(
            id=114,
            vertex1=_vertex_prototypes[56],
            vertex2=_vertex_prototypes[86],
        ),
        stk.Edge(
            id=115,
            vertex1=_vertex_prototypes[57],
            vertex2=_vertex_prototypes[86],
        ),
        stk.Edge(
            id=116,
            vertex1=_vertex_prototypes[39],
            vertex2=_vertex_prototypes[89],
        ),
        stk.Edge(
            id=117,
            vertex1=_vertex_prototypes[40],
            vertex2=_vertex_prototypes[89],
        ),
        stk.Edge(
            id=118,
            vertex1=_vertex_prototypes[87],
            vertex2=_vertex_prototypes[89],
        ),
        stk.Edge(
            id=119,
            vertex1=_vertex_prototypes[88],
            vertex2=_vertex_prototypes[89],
        ),
    )
