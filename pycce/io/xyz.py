import warnings

import numpy as np
from pycce.bath.array import BathArray

from .base import set_isotopes


# hbar mu0 /4pi I have no idea about units, from mengs code
# UPDATE: derived, it checks out
# HBAR = 1.054571729
# External HF given in MHz, transform to kHz * rad
# MHZ_TO_KHZ = 2 * np.pi * 1000


def read_xyz(xyz, skiprows: int = 2, spin_types=None, isotopes=None, imap=None):
    """
    Read positions of bath spins from xyz file.

    If xyz contains names of the elements and not isotopes, changes the names to the most abundant nonzero spin
    isotopes.

    Args:
        xyz (ndarray or str):
            Either:

            - ndarray with dtype containing [('N', np.unicode_, 16), ('xyz', np.float64, (3,))], which
              contains types and positions of bath spins.
            - ndarray with coordinates of the spins.
            - Name of the xyz-file containing isotope name and xyz coordinates.

        skiprows (int): used when xyz is filename. Number of rows to skip in the file
        spin_types (SpinDict or list of tuples): SpinDict containing SpinTypes of isotopes or input to make one.
        isotopes (dict): Optional.
            Dictionary with entries: {"element" : "isotope"}, where "element" is the name of the element
            in DFT output, "isotope" is the name of the isotope.

    Returns:
        BathArray: Array of atoms read from xyz argument.

    """

    if isinstance(xyz, BathArray):
        atoms = xyz.copy()
        if spin_types is not None:
            try:
                atoms.add_type(**spin_types)
            except TypeError:
                atoms.add_type(*spin_types)
        if imap is not None:
            if atoms.imap is None:
                atoms.imap = imap
            else:
                atoms.imap = atoms.imap + imap
        return atoms

    elif isinstance(xyz, np.ndarray):
        dataset = xyz
    else:
        dt_read = np.dtype([('N', np.unicode_, 16), ('xyz', np.float64, (3,))])
        dataset = np.loadtxt(xyz, dtype=dt_read, skiprows=skiprows)

    with warnings.catch_warnings(record=True) as w:
        atoms = BathArray(array=dataset, types=spin_types, imap=imap)
    if w:
        set_isotopes(atoms, isotopes=isotopes)

    return atoms
