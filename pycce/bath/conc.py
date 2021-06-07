# Dictionary with concentrations of common isotopes (old)
from .array import SpinType, SpinDict
_common_concentrations = {
    'H':  {'1H': 0.999885, '2H': 0.000115, '3H': 0},
    'He': {'3He': 1.34e-6},
    'Li': {'6Li': 0.07594, '7Li': 0.9241},
    'Be': {'9Be': 1},
    'B':  {'10B': 0.199, '11B': 0.801},
    'C':  {'13C': 0.0107},
    'N':  {'14N': 0.99636, '15N': 0.00364},
    'O':  {'17O': 3.8e-4},
    'F':  {'19F': 1},
    'Ne': {'21Ne': 0.0027},
    'Na': {'23Na': 1},
    'Mg': {'25Mg': 0.1},
    'Al': {'27Al': 1},
    'Si': {'29Si': 0.04685},
    'P':  {'31P': 1},
    'S':  {'33S': 0.0075},
    'Cl': {'35Cl': 0.7576, '37Cl': 0.2424},
    'K':  {'39K': 0.932581, '41K': 0.067302},
    'Ca': {'43Ca': 0.00135},
    'Sc': {'45Sc': 1},
    'Ti': {'47Ti': 0.0744, '49Ti': 0.0541},
    'V':  {'50V': 0.00250, '51V': 0.99750},
    'Cr': {'53Cr': 0.09501},
    'Mn': {'55Mn': 1},
    'Fe': {'57Fe': 0.02119},
    'Co': {'59Co': 1},
    'Ni': {'61Ni': 0.011399},
    'Cu': {'63Cu': 0.6915, '65Cu': 0.3085},
    'Zn': {'67Zn': 0.0404},
    'Ga': {'69Ga': 0.60108, '71Ga': 0.39892},
    'Ge': {'73Ge': 0.0776},
    'As': {'75As': 1},
    'Se': {'77Se': 0.076},
    'Br': {'79Br': 0.5069, '81Br': 0.4931},
    'Kr': {'83Kr': 0.115},
    'Rb': {'85Rb': 0.7217, '87Rb': 0.2783},
    'Sr': {'87Sr': 0.07},
    'Y':  {'89Y': 1},
    'Zr': {'91Zr': 0.1122},
    'Nb': {'93Nb': 1},
    'Mo': {'95Mo': 0.15873, '97Mo': 0.09582},
    'Ru': {'99Ru': 0.1276, '101Ru': 0.1706},
    'Rh': {'103Rh': 1},
    'Pd': {'105Pd': 0.2733},
    'Ag': {'107Ag': 0.51839, '109Ag': 0.48161},
    'Cd': {'111Cd': 0.12795, '113Cd': 0.12227},
    'In': {'113In': 0.04281, '115In': 0.95719},
    'Sn': {'115Sn': 0.0034, '117Sn': 0.0768, '119Sn': 0.0859},
    'Sb': {'121Sb': 0.5721, '123Sb': 0.4279},
    'Te': {'123Te': 0.0089, '125Te': 0.0707},
    'I':  {'127I': 1},
    'Xe': {'129Xe': 0.264006, '131Xe': 0.212324},
    'Cs': {'133Cs': 1},
    'Ba': {'135Ba': 0.06592, '137Ba': 0.11232},
    'La': {'138La': 0.0008881, '139La': 0.9991119},
    'Pr': {'141Pr': 1},
    'Nd': {'143Nd': 0.12174, '145Nd': 0.08293},
    'Sm': {'147Sm': 0.1499, '149Sm': 0.1382},
    'Eu': {'151Eu': 0.4781, '153Eu': 0.5219},
    'Gd': {'155Gd': 0.148, '157Gd': 0.1565},
    'Tb': {'159Tb': 1},
    'Dy': {'161Dy': 0.18889, '163Dy': 0.24896},
    'Ho': {'165Ho': 1},
    'Er': {'167Er': 0.22869},
    'Tm': {'169Tm': 1},
    'Yb': {'171Yb': 0.1409, '173Yb': 0.16103},
    'Lu': {'175Lu': 0.97401, '176Lu': 0.02599},
    'Hf': {'177Hf': 0.1860, '179Hf': 0.1362},
    'Ta': {'181Ta': 0.9998799},
    'W':  {'183W': 0.1431},
    'Re': {'185Re': 0.374, '187Re': 0.6260},
    'Os': {'187Os': 0.0196, '189Os': 0.1615},
    'Ir': {'191Ir': 0.373, '193Ir': 0.627},
    'Pt': {'195Pt': 0.3378},
    'Au': {'197Au': 1},
    'Hg': {'199Hg': 0.16938, '201Hg': 0.13170},
    'Tl': {'203Tl': 0.2952, '205Tl': 0.7048},
    'Pb': {'207Pb': 0.221},
    'Bi': {'209Bi': 1},
    'U':  {'235U': 0.007204}
}

_common_isotopes = SpinDict()
# H
_common_isotopes['1H'] = SpinType('1H', 1 / 2, 26.7519, 0)
_common_isotopes['2H'] = SpinType('2H', 1, 4.1066, 0.00286)
_common_isotopes['3H'] = SpinType('3H', 1 / 2, 28.535, 0)

# He
_common_isotopes['3He'] = SpinType('3He', 1 / 2, -20.38, 0)

# Li
_common_isotopes['6Li'] = SpinType('6Li', 1, 3.9371, -0.000806)
_common_isotopes['7Li'] = SpinType('7Li', 3 / 2, 10.3976, -0.0400)

# Be
_common_isotopes['9Be'] = SpinType('9Be', 3 / 2, -3.759666, 0.0529)

# mfield
_common_isotopes['10B'] = SpinType('10B', 3, 2.875, 0.0845)
_common_isotopes['11B'] = SpinType('11B', 3 / 2, 8.584, 0.04059)

# C
_common_isotopes['13C'] = SpinType('13C', 1 / 2, 6.7283, 0)

# N
_common_isotopes['14N'] = SpinType('14N', 1, 1.9338, 0.02044)
_common_isotopes['15N'] = SpinType('15N', 1 / 2, -2.712, 0)

# O
_common_isotopes['17O'] = SpinType('17O', 5 / 2, -3.6279, -0.0256)

# F
_common_isotopes['19F'] = SpinType('19F', 1 / 2, 25.181, 0)

# Ne
_common_isotopes['21Ne'] = SpinType('21Ne', 3 / 2, -2.113, 0.102)

# Na
_common_isotopes['23Na'] = SpinType('23Na', 3 / 2, 7.0801, 0.104)

# Mg
_common_isotopes['25Mg'] = SpinType('25Mg', 5 / 2, -1.639, 0.199)

# Al
_common_isotopes['27Al'] = SpinType('27Al', 5 / 2, 6.976, 0.1466)

# Si
_common_isotopes['29Si'] = SpinType('29Si', 1 / 2, -5.3188, 0)

# P
_common_isotopes['31P'] = SpinType('31P', 1 / 2, 10.841, 0)

# S
_common_isotopes['33S'] = SpinType('33S', 3 / 2, 2.055, -0.0678)

# Cl
_common_isotopes['35Cl'] = SpinType('35Cl', 3 / 2, 2.624, -0.0817)
_common_isotopes['37Cl'] = SpinType('37Cl', 3 / 2, 2.1842, -0.0644)

# K
_common_isotopes['39K'] = SpinType('39K', 3 / 2, 1.2498, 0.0585)
_common_isotopes['40K'] = SpinType('40K', 4, -1.554285388471735, -0.073)
_common_isotopes['41K'] = SpinType('41K', 3 / 2, 0.686, 0.0711)

# Ca
_common_isotopes['43Ca'] = SpinType('43Ca', 7 / 2, -1.8025, -0.0408)

# Sc
_common_isotopes['45Sc'] = SpinType('45Sc', 7 / 2, 6.5081, -0.220)

# Ti
_common_isotopes['47Ti'] = SpinType('47Ti', 5 / 2, -1.5105, 0.302)
_common_isotopes['49Ti'] = SpinType('49Ti', 7 / 2, -1.5109, 0.247)

# V
_common_isotopes['50V'] = SpinType('50V', 6, 2.6717, 0.21)
_common_isotopes['51V'] = SpinType('51V', 7 / 2, 7.0453, -0.043)

# Cr
_common_isotopes['53Cr'] = SpinType('53Cr', 3 / 2, -1.512, 0.15)

# Mn
_common_isotopes['55Mn'] = SpinType('55Mn', 5 / 2, 6.608, 0.330)

# Fe
_common_isotopes['57Fe'] = SpinType('57Fe', 1 / 2, 0.8661, 0)

# Co
_common_isotopes['59Co'] = SpinType('59Co', 7 / 2, 6.317, 0.42)

# Ni
_common_isotopes['61Ni'] = SpinType('61Ni', 3 / 2, -2.394, 0.162)

# Cu
_common_isotopes['63Cu'] = SpinType('63Cu', 3 / 2, 7.0974, -0.220)
_common_isotopes['65Cu'] = SpinType('65Cu', 3 / 2, 7.6031, -0.204)

# Zn
_common_isotopes['67Zn'] = SpinType('67Zn', 5 / 2, 1.6768, 0.150)

# Ga
_common_isotopes['69Ga'] = SpinType('69Ga', 3 / 2, 6.4323, 0.171)
_common_isotopes['71Ga'] = SpinType('71Ga', 3 / 2, 8.1731, 0.107)

# Ge
_common_isotopes['73Ge'] = SpinType('73Ge', 9 / 2, -0.9357, -0.196)

# As
_common_isotopes['75As'] = SpinType('75As', 3 / 2, 4.595, 0.314)

# Se
_common_isotopes['77Se'] = SpinType('77Se', 1 / 2, 5.12, 0)

# Br
_common_isotopes['79Br'] = SpinType('79Br', 3 / 2, 6.7228, 0.313)
_common_isotopes['81Br'] = SpinType('81Br', 3 / 2, 7.2468, 0.262)

# Kr
_common_isotopes['83Kr'] = SpinType('83Kr', 9 / 2, -1.033, 0.259)

# Rb
_common_isotopes['85Rb'] = SpinType('85Rb', 5 / 2, 2.5909, 0.276)
_common_isotopes['87Rb'] = SpinType('87Rb', 3 / 2, 8.7807, 0.1335)

# Sr
_common_isotopes['87Sr'] = SpinType('87Sr', 9 / 2, -1.163, 0.305)

# Y
_common_isotopes['89Y'] = SpinType('89Y', 1 / 2, -1.3155, 0)

# Zr
_common_isotopes['91Zr'] = SpinType('91Zr', 5 / 2, -2.4959, -0.176)

# Nb
_common_isotopes['93Nb'] = SpinType('93Nb', 9 / 2, 6.564, -0.32)

# Mo
_common_isotopes['95Mo'] = SpinType('95Mo', 5 / 2, -1.75, -0.022)
_common_isotopes['97Mo'] = SpinType('97Mo', 5 / 2, -1.787, 0.255)

# Ru
_common_isotopes['99Ru'] = SpinType('99Ru', 3 / 2, -1.234, 0.079)
_common_isotopes['101Ru'] = SpinType('101Ru', 5 / 2, -1.383, 0.46)

# Rh
_common_isotopes['103Rh'] = SpinType('103Rh', 1 / 2, -0.846, 0)

# Pd
_common_isotopes['105Pd'] = SpinType('105Pd', 5 / 2, -1.2305, 0.66)

# Ag
_common_isotopes['107Ag'] = SpinType('107Ag', 1 / 2, -1.087, 0)
_common_isotopes['109Ag'] = SpinType('109Ag', 1 / 2, -1.25, 0)

# Cd
_common_isotopes['111Cd'] = SpinType('111Cd', 1 / 2, -5.6926, 0)
_common_isotopes['113Cd'] = SpinType('113Cd', 1 / 2, -5.955, 0)

# In
_common_isotopes['113In'] = SpinType('113In', 9 / 2, 5.8782, 0.759)
_common_isotopes['115In'] = SpinType('115In', 9 / 2, 5.8908, 0.770)

# Sn
_common_isotopes['115Sn'] = SpinType('115Sn', 1 / 2, -8.8014, 0)
_common_isotopes['117Sn'] = SpinType('117Sn', 1 / 2, -9.589, 0)
_common_isotopes['119Sn'] = SpinType('119Sn', 1 / 2, -10.0138, 0)

# Sb
_common_isotopes['121Sb'] = SpinType('121Sb', 5 / 2, 6.4355, -0.543)
_common_isotopes['123Sb'] = SpinType('123Sb', 7 / 2, 3.4848, -0.692)

# Te
_common_isotopes['123Te'] = SpinType('123Te', 1 / 2, -7.049, 0)
_common_isotopes['125Te'] = SpinType('125Te', 1 / 2, -8.498, 0)

# I
_common_isotopes['127I'] = SpinType('127I', 5 / 2, 5.3817, -0.696)

# Xe
_common_isotopes['129Xe'] = SpinType('129Xe', 1 / 2, -7.441, 0)
_common_isotopes['131Xe'] = SpinType('131Xe', 3 / 2, 2.206, -0.114)

# Cs
_common_isotopes['133Cs'] = SpinType('133Cs', 7 / 2, 3.5277, -0.00343)

# Ba
_common_isotopes['135Ba'] = SpinType('135Ba', 3 / 2, 2.671, 0.160)
_common_isotopes['137Ba'] = SpinType('137Ba', 3 / 2, 2.988, 0.245)

# La
_common_isotopes['138La'] = SpinType('138La', 5, 3.5575, 0.21)
_common_isotopes['139La'] = SpinType('139La', 7 / 2, 3.8085, 0.200)

# Pr
_common_isotopes['141Pr'] = SpinType('141Pr', 5 / 2, 8.190, -0.077)

# Nd
_common_isotopes['143Nd'] = SpinType('143Nd', 7 / 2, -1.45735, -0.61)
_common_isotopes['145Nd'] = SpinType('145Nd', 7 / 2, -0.89767, -0.314)

# Sm
_common_isotopes['147Sm'] = SpinType('147Sm', 7 / 2, -1.111, -0.27)
_common_isotopes['149Sm'] = SpinType('149Sm', 7 / 2, -0.91368, 0.075)

# Eu
_common_isotopes['151Eu'] = SpinType('151Eu', 5 / 2, 6.650967, 0.95)
_common_isotopes['153Eu'] = SpinType('153Eu', 5 / 2, 2.93572, 2.28)

# Gd
_common_isotopes['155Gd'] = SpinType('155Gd', 3 / 2, -0.821225, 1.27)
_common_isotopes['157Gd'] = SpinType('157Gd', 3 / 2, -1.0850, 1.35)

# Tb
_common_isotopes['159Tb'] = SpinType('159Tb', 3 / 2, 6.43219, 1.432)

# Dy
_common_isotopes['161Dy'] = SpinType('161Dy', 5 / 2, -0.919568, 2.51)
_common_isotopes['163Dy'] = SpinType('163Dy', 5 / 2, 1.28931, 2.318)

# Ho
_common_isotopes['165Ho'] = SpinType('165Ho', 7 / 2, 7.988746715879683, 3.58)

# Er
_common_isotopes['167Er'] = SpinType('167Er', 7 / 2, -0.771575, 3.57)

# Tm
_common_isotopes['169Tm'] = SpinType('169Tm', 1 / 2, -2.21, 0)

# Yb
_common_isotopes['171Yb'] = SpinType('171Yb', 1 / 2, 4.7248, 0)
_common_isotopes['173Yb'] = SpinType('173Yb', 5 / 2, -1.2414, 2.80)

# Lu
_common_isotopes['175Lu'] = SpinType('175Lu', 7 / 2, 3.05469, 3.49)
_common_isotopes['176Lu'] = SpinType('176Lu', 7, 2.163448, 4.92)

# Hf
_common_isotopes['177Hf'] = SpinType('177Hf', 7 / 2, 1.081, 3.37)
_common_isotopes['179Hf'] = SpinType('179Hf', 9 / 2, -0.679, 3.79)

# Ta

_common_isotopes['180Ta'] = SpinType('180Ta', 9, 2.56761, 4.8)
_common_isotopes['181Ta'] = SpinType('181Ta', 7 / 2, 3.22, 3.17)

# W
_common_isotopes['183W'] = SpinType('183W', 1 / 2, 1.12, 0)

# Re
_common_isotopes['185Re'] = SpinType('185Re', 5 / 2, 6.105548, 2.18)
_common_isotopes['187Re'] = SpinType('187Re', 5 / 2, 6.1682895, 2.07)

# Os
_common_isotopes['187Os'] = SpinType('187Os', 1 / 2, 0.616, 0)
_common_isotopes['189Os'] = SpinType('189Os', 3 / 2, 2.10713, 0.86)

# Ir
_common_isotopes['191Ir'] = SpinType('191Ir', 3 / 2, 0.4643, 0.816)
_common_isotopes['193Ir'] = SpinType('193Ir', 3 / 2, 0.5054, 0.751)

# Pt
_common_isotopes['195Pt'] = SpinType('195Pt', 1 / 2, 5.8383, 0)

# Au
_common_isotopes['197Au'] = SpinType('197Au', 3 / 2, 0.4625, 0.547)

# Hg
_common_isotopes['199Hg'] = SpinType('199Hg', 1 / 2, 4.84579, 0)
_common_isotopes['201Hg'] = SpinType('201Hg', 3 / 2, -1.7776, 0.387)

# Tl
_common_isotopes['203Tl'] = SpinType('203Tl', 1 / 2, 15.539, 0)
_common_isotopes['205Tl'] = SpinType('205Tl', 1 / 2, 15.6921840, 0)

# Pb
_common_isotopes['207Pb'] = SpinType('207Pb', 1 / 2, 5.676, 0)

# Bi
_common_isotopes['209Bi'] = SpinType('209Bi', 9 / 2, 4.37465, -0.516)

# U
_common_isotopes['235U'] = SpinType('235U', 7 / 2, -0.52, 4.936)

