#! /usr/bin/env python

__author__ = "John Bonini"

import numpy as np
from math import floor
import os
import re
import logging
import datetime
import zlib
from matplotlib import pyplot as plt
import six
from glob import glob
from collections import MutableSequence
import pymatgen as pmg
from pymatgen.electronic_structure.bandstructure import Kpoint
from itertools import izip
from monty.json import MSONable, jsanitize
from pymatgen.apps.borg.hive import AbstractDrone, VaspToComputedEntryDrone
from matgendb.creator import VaspToDbTaskDrone
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymongo import MongoClient

BOHR_TO_ANG =  0.52917721067
HARTREE_TO_EV = 27.21138387
logger = logging.getLogger(__name__)

class DistortionMode(MSONable):
    """
    Contains the frequency, q vector, displacement patterns and whether or not the mode is soft

    Args:
        frequency:
            frequency of the mode as a float
        soft:
            whether or not the mode is soft as a boolean
        q_vec:
            q vector associated with the mode as a Qpoint object
        displacements:
            list of displacements and weights cooresponding to the largest supercell (given in grid.dat) as a numpy array
    """
    def __init__(self, frequency, soft, q_vec, displacements):
        self._frequency = frequency
        self._soft = soft
        self._q_vec = q_vec
        self._displacements = displacements

    def get_distorted_structure(self, super_equilibrium, reduce_to_primitive=True, tolerance=0.001):
        #TODO: Tune tolerance parameter default
        """
        Method returning a pymatgen structure cooresponding to this distortion:

        Args:
            super_equilibrium:
                The largest supercell structure with no displacements as a pymatgen structure
            reduce_to_primitive:
                Whether or not to reduce the resulting structure to it's primitive cell (True by default)
            tolerance:
                Tolerance to use if reduce_to_primitive is True (0.25 by default)

        Returns:
            a pymatgen structure object cooresponding to this distortion
        """
        distStruct = pmg.Structure.from_dict(super_equilibrium.as_dict())
        for i,disp in enumerate(self.displacements):
            distStruct.translate_sites(i, disp[:3]*BOHR_TO_ANG, frac_coords=False)
        if reduce_to_primitive:
            return distStruct.get_primitive_structure(tolerance=tolerance)
        return distStruct

    def __repr__(self):
        outs = ["Frequency :\t"+str(self.frequency),
                ' '+str(self.q_vec.frac_coords).strip('[').strip(']'),
                'Displacement pattern for each atom:']
        if self.soft:
            outs[0]+=' (SOFT)'
        for disp in self.displacements:
            outs.append(' '+str(disp).strip('[').strip(']'))
        return "\n".join(outs)

    @property
    def frequency(self):
        return self._frequency

    @property
    def soft(self):
        return self._soft

    @property
    def q_vec(self):
        return self._q_vec

    @property
    def displacements(self):
        return self._displacements

    @classmethod
    def from_dict(cls, d):
        return cls(d["frequency"], d["soft"], Qpoint.from_dict(d["q_vec"]), np.array(d["displacements"]))

    def as_dict(self):
        return jsanitize(super(DistortionMode, self).as_dict(),strict=True)

    def get_scaled_displacements(self, scale):
        return scale*self.displacements

class DispPatterns(MutableSequence, MSONable):
    """
    Args:
        dispPatterns:
            a list of DistortionMode objects
    """
    def __init__(self, dispPatterns):
        """modes is a list of DistortionMode objects"""
        self._dispPatterns = dispPatterns

    @property
    def dispPatterns(self):
        return self._dispPatterns

    def __repr__(self):
        outs = ['disp_patterns.dat']
        for s in self:
            outs.append(repr(s))
        return "\n".join(outs)

    @classmethod
    def from_files(cls, lte_dir):
        """
        Generate from caesars disp_patterns.dat file
        
        Args:
            lte_dir:
                location of caesar's lte directory

        Returns:
            DispPatterns object
        """
        with open(os.path.join(lte_dir, 'disp_patterns.dat')) as f:
            matches = re.findall(r'\s*Frequency\s*:\s*([-\d.E]*)\s*((?:\(SOFT\))?)\n\s*([-\d.E ]*)\n\s*Displacement pattern for each atom:\s*\n([-\d.E\s]*)\n\n', f.read())
        BZ = BZPoints.from_files(lte_dir, to_unit_cell=True)
        modes = [DistortionMode(
            float(m[0]),
            m[1]=='(SOFT)',
            BZ.match_q_point(np.fromstring(m[2], sep=' '), coords_are_cartesian=True),
            np.array([np.fromstring(rawdisp, sep=' ') for rawdisp in m[3].split('\n')])
            ) for m in matches]
        return cls(modes)

    def __getitem__(self, ind):
        return self.dispPatterns[ind]

    def __len__(self):
        return len(self.dispPatterns)

    def __delitem__(self, i):
        self._dispPatterns.__delitem__(i)

    def insert(self, i, mode):
        if isinstance(mode, DistortionMode):
            self._dispPatterns.insert(i, mode)
        else:
            raise TypeError("Elements of DispPatterns must be DistortionMode objects")

    def __setitem__(self, i, mode):
        if isinstance(mode, DistortionMode):
            self._dispPatterns[i] = mode
        else:
            raise TypeError("Elements of DispPatterns must be DistortionMode objects")

    @classmethod
    def from_dict(cls, d):
        return cls([DistortionMode.from_dict(m) for m in d["dispPatterns"]])

    def as_dict(self):
        return jsanitize(super(DispPatterns, self).as_dict(),strict=True)

class FreqDos(MSONable):
    """
    Phonon Density of states
    Args:
        dos:
            density of states as a numpy array
    """
    def __init__(self, dos):
        self.dos = dos

    @classmethod
    def from_files(cls, lte_dir):
        """
        Args:
            lte_dir:
                location of caesar's lte directory

        Returns:
            FreqDos object
        """
        data = np.genfromtxt(os.path.join(lte_dir, 'freq_dos.dat'))
        data[:,0] = data[:,0]*HARTREE_TO_EV
        return cls(data)

    def __repr__(self):
        return repr(self.dos)

    def plot(self, show=True):
        """
        Plot the density of states using matplotlib
        Args:
            show:
                whether or not to show the plot (default is True)
        Returns:
            a matplotlib figure object with a plot of the DOS
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.dos[:,0], self.dos[:,1])
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Density of states')
        if show:
            fig.show()
        return fig

    def as_dict(self):
        return jsanitize(super(FreqDos, self).as_dict(),strict=True)

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d['dos']))

class Energies(MSONable):
    """
    Various energies caesar calculates
    Each is given as an array of the form [[temp1,energy1], [temp2,energy2]...]
    Temperatures are in Kelvin
    Energies are in Hartree

    Args:
        grid:
            grid energy
        interpolated:
            interpolated energy
        grid_free:
            grid free energy
        interpolated_free:
            interpolated free energy
    """
    def __init__(self, grid, interpolated, grid_free, interpolated_free):
        self.grid = grid
        self.interpolated = interpolated
        self.grid_free = grid_free
        self.interpolated_free = interpolated_free

    @classmethod
    def from_files(cls, lte_dir):
        """
        Args:
            lte_dir:
                location of caesar's lte directory
        Returns:
            Energies object
        """
        grid = cls.readFile(os.path.join(lte_dir, 'grid_energy.dat'))
        interpolated = cls.readFile(os.path.join(lte_dir, 'interpolated_energy.dat'))
        grid_free = cls.readFile(os.path.join(lte_dir, 'grid_free_energy.dat'))
        interpolated_free = cls.readFile(os.path.join(lte_dir, 'interpolated_free_energy.dat'))
        return cls(grid, interpolated, grid_free, interpolated_free)

    @staticmethod
    def readFile(dat):
        s = np.genfromtxt(dat)[:,0:2]   #only taking first two columns, third is the second converted to from Ha to eV
        return s.tolist()

    def __repr__(self):
        outs = [
                'grid : '+str(self.grid[0])+'\n...',
                'interpolated : '+str(self.interpolated[0])+'\n...',
                'grid_free : '+str(self.grid_free[0])+'\n...',
                'interpolated_free : '+str(self.interpolated_free[0])+'\n...',
                ]
        return "\n".join(outs)

class Qpoint(Kpoint):
    """
    Extends pymatgen Kpoint to also contain multiplicity in BZ (due to the structure's symmetry) and the index of the supercell associated with this Qpoint

    Args:
        coords:
            coordinates of the q-point as a numpy array
        lattice:
            represents the reciprocal lattice as a pymatgen Lattice object
        multiplicity:
            multiplicity of the q-point due to crystal symmetry
        supercell_i:
            index of the computed supercell corresponding to this q-point.
    """
    def __init__(self, coords, lattice, multiplicity, supercell_i, **kwargs):
        self._multiplicity = multiplicity
        self._supercell_i = supercell_i
        super(Qpoint, self).__init__(coords, lattice, **kwargs)

    @property
    def multiplicity(self):
        return self._multiplicity

    @property
    def supercell_i(self):
        return self._supercell_i

    def __repr__(self):
        return str(self) +'\n multiplicity : '+str(self.multiplicity)+'\n supercell index : '+str(self.supercell_i)

    def as_dict(self):
        d = super(Qpoint, self).as_dict()
        d["multiplicity"] = self.multiplicity
        d["supercell_i"] = self.supercell_i
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d["fcoords"], pmg.Lattice.from_dict(d["lattice"]), d["multiplicity"], d["supercell_i"], label=d["label"])

class BZPoints(MutableSequence, MSONable):
    """
    Set of q-points in the Brillouin Zone

    Args:
        q_points: 
            list of Qpoint objects
        symmetry: 
            symmetry of the crystal structure as a list of tuples of numpy arrays in the form (rot_matrix, translation_vector)
    """
    def __init__(self, q_points, symmetry):
        if len(set([qpt.lattice for qpt in q_points]))==1:
            self._q_points = q_points
        else:
            raise ValueError("q_points must all coorespond to the same lattice")
        self._symmetry = symmetry

    def match_q_point(self, pt, coords_are_cartesian=False):
        """
        Method which given a point returns a Qpoint object with appropriate label, multiplicity, and supercell_i if BZpoints has this information

        Args:
            pt:
                a point in reciprocal space as a numpy array
            coords_are_cartesian:
                False if using reduced coordinates, True if using cartesian coordinates (default is False)

            
        Returns:
            A Qpoint object
        """
        this_pt = self.lattice.get_fractional_coords((1/BOHR_TO_ANG)*np.array(pt)) if coords_are_cartesian else pt
        for qpt in self:
            if (np.abs(qpt.frac_coords - np.array([p - floor(p) for p in this_pt])) < 1e-5).all():
                return qpt
            if qpt.multiplicity > 1:
                for sym_op in self.symmetry:
                    if (np.abs(qpt.frac_coords - np.array([p - floor(p) for p in (np.dot(sym_op[0], this_pt) + sym_op[1])])) < 1e-5).all():
                        return Qpoint(np.array([p - floor(p) for p in this_pt]), self.lattice, qpt.multiplicity, qpt.supercell_i, label = qpt.label)
        #Should probably throw a warning here
        return Qpoint(this_pt, self.lattice, None, None)


    @property
    def q_points(self):
        return self._q_points

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def lattice(self):
        return self.q_points[0].lattice

    @classmethod
    def from_files(cls, lte_dir, **kwargs):
        """
        Args:
            lte_dir:
                location of caesar's lte directory

        Returns:
            BZPoints object
        """
        if 'to_unit_cell' not in kwargs.keys():
            kwargs['to_unit_cell']=True
        ibz = np.genfromtxt(os.path.join(lte_dir, 'ibz.dat'))
        kpt2sc = np.genfromtxt(os.path.join(lte_dir, 'kpoint_to_supercell.dat'))
        latdat = np.genfromtxt(os.path.join(lte_dir, 'lattice.dat'))
        real_lattice = pmg.Lattice(BOHR_TO_ANG*latdat)
        lattice = real_lattice.reciprocal_lattice

        symmetrydat = np.genfromtxt(os.path.join(lte_dir, 'symmetry.dat'), skip_header=1)
        symmetry = [ (np.array([m1,m2,m3]), t) for m1,m2,m3,t in izip(*[iter(symmetrydat)]*4) ]

        q_points = []
        for mult, sc in zip(ibz, kpt2sc):
            if (np.abs(mult[:3] - sc[:3])<1E-5).all():
                q_points.append(Qpoint(mult[:3], lattice, mult[3], sc[3]-1, **kwargs))
            else:
                raise ValueError("Q points in ibz.dat and kpoint_tosupercell.dat must be the same and in the same order")
        return cls(q_points, symmetry)

    def __getitem__(self, ind):
        return self.q_points[ind]

    def __len__(self):
        return len(self.q_points)

    def __delitem__(self, i):
        self._q_points.__delitem__(i)

    def insert(self, i, mode):
        if isinstance(mode, Qpoint):
            self._q_points.insert(i, mode)
        else:
            raise TypeError("Elements of BZPoints must be Qpoint objects")

    def __setitem__(self, i, mode):
        if isinstance(mode, Qpoint):
            self._q_points[i] = mode
        else:
            raise TypeError("Elements of BZPoints must be Qpoint objects")

    def as_dict(self):
        return jsanitize(super(BZPoints, self).as_dict(),strict=True)

    @classmethod
    def from_dict(cls, d):
        return cls([Qpoint.from_dict(q) for q in d['q_points']], [(np.array(s[0]), np.array(s[1])) for s in d['symmetry']])

class CaesarInputs(MSONable):
    """
    Contains various information input in to caesar

    Args:
        starting_structure:
            Initial structure as a pymatgen Structure object
        grid:
            grid of q-points for calculation as a numpy array
        symmetry:
            symmetry of the crystal structure as a list of tuples of numpy arrays in the form (rot_matrix, translation_vector)
        spins:
            ordered list of spins corresponding to each atom as a numpy array
        masses:
            ordered list of atomic masses corresponding to each atom as a numpy array
    """
    def __init__(self, starting_structure, grid, symmetry, spins, masses):
        self._starting_structure = starting_structure
        self._grid = grid
        self._symmetry = symmetry
        self._spins = spins
        self._masses = masses

    @property
    def starting_structure(self):
        return self._starting_structure

    @property
    def grid(self):
        return self._grid

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def spins(self):
        return self._spins

    @property
    def masses(self):
        return self._masses

    @classmethod
    def from_files(cls, lte_dir):
        """
        Args:
            lte_dir:
                location of caesar's lte directory

        Returns:
            CaesarInputs object
        """
        grid = np.genfromtxt(os.path.join(lte_dir, 'grid.dat'))
        symmetrydat = np.genfromtxt(os.path.join(lte_dir, 'symmetry.dat'), skip_header=1)
        symmetry = [ (np.array([m1,m2,m3]), t) for m1,m2,m3,t in izip(*[iter(symmetrydat)]*4) ]
        structure, masses, spins = readCaesarEquilibriumStructure(os.path.join(lte_dir, 'equilibrium.dat'), os.path.join(lte_dir, 'lattice.dat'))
        return cls(structure, grid, symmetry, spins, masses)

    def as_dict(self):
        return jsanitize(super(CaesarInputs, self).as_dict(),strict=True)

    @classmethod
    def from_dict(cls, d):
        return cls(pmg.Structure.from_dict(d['starting_structure']), np.array(d['grid']), [(np.array(s[0]), np.array(s[1])) for s in d['symmetry']], d['spins'], d['masses'])

    def get_super_equilibrium(self):
        tmpStruct = pmg.Structure.from_dict(self.starting_structure.as_dict())
        tmpStruct.make_supercell(self.grid)
        return tmpStruct

class DynMat(MutableSequence, MSONable):
    def __init__(self, dynmats):
        """
        dynmats is a list of arrays containing dynamical matrix information as recorded by caesar
        Args:
            dynmats:
                list of numpy arrays each corresponding to a supercell used in the calculation
        """
        self.dynmats = dynmats

    @classmethod
    def from_files(cls, lte_dir):
        """
        Args:
            lte_dir:
                location of caesar's lte directory

        Returns:
            DynMat object
        """
        files = sorted(glob(os.path.join(lte_dir, 'dyn_mat.*.dat')))
        dynmat_dat = [(os.path.split(dynmat)[-1].split('.')[-2], np.genfromtxt(dynmat)) for dynmat in files]
        dynmats = []
        for i,d in dynmat_dat:
            this_mat = np.zeros((int(d[:,0].max()), int(d[:,1].max()), int(d[:,2].max()), int(d[:,3].max())))
            for line in d:
                this_mat[int(line[0])-1, int(line[1])-1, int(line[2])-1, int(line[3])-1] = line[4]+1.j*line[5]
            dynmats.append((i, this_mat))
        return cls(dynmats)

    def __getitem__(self, ind):
        return self.dynmats[ind]

    def __len__(self):
        return len(self.dynmats)

    def __delitem__(self, i):
        self._dynmats.__delitem__(i)

    def insert(self, i, dynmat):
        self._dynmats.insert(i, dynmat)

    def __setitem__(self, i, dynmat):
        self._dynmats[i] = dynmat

    def as_dict(self):
        return jsanitize(super(DynMat, self).as_dict(),strict=True)

    @classmethod
    def from_dict(cls, d):
        return cls([(i,np.array(dm)) for i,dm in d["dynmats"]])

class PhononDispersion(MSONable):
    """
    Args:
        dispersion:
            dispersion curves as a numpy array
        path:
            List of tuples of (Qpoint, x) where x is distance traveled along the path to get to Qpoint
    """
    def __init__(self, dispersion, path):
        self.dispersion = dispersion
        self.path = path

    @classmethod
    def from_files(cls, lte_dir):
        """
        Args:
            lte_dir:
                location of caesar's lte directory

        Returns:
            PhononDispersion object
        """
        disp = np.genfromtxt(os.path.join(lte_dir, 'phonon_dispersion_curve.dat'))
        disp[:,1:] = HARTREE_TO_EV*disp[:,1:]
        high_symdat = np.genfromtxt(os.path.join(lte_dir, 'high_symmetry_points.dat'))
        BZ = BZPoints.from_files(lte_dir, to_unit_cell=True)

        pathdat = []
        with open(os.path.join(lte_dir, 'path.dat')) as f:
            for line in f:
                #Check that there is actually a label in the file before adding it
                if len(line.split())>3:
                    pathdat.append((np.array(line.split()[:3], dtype=float),line.split()[-1]))
                else:
                    pathdat.append((np.array(line.split()[:3], dtype=float),None))
        path = []
        for pt, x in zip(pathdat, high_symdat[:,1]):
            qpt = BZ.match_q_point(pt[0])
            qpt._label = pt[1]
            path.append((qpt,x))
        return cls(disp, path)

    def plot(self, show=True):
        """
        Plot the dispersion curves using matplotlib
        Args:
            show:
                whether or not to show the plot (default is True)
        Returns:
            a matplotlib figure object with a plot of the dispersion curve
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(self.dispersion[:,0].min(), self.dispersion[:,0].max())
        ax.set_xlabel('k-space')
        ax.set_ylabel('Energy (eV)')
        ax.plot(self.dispersion[:,0], self.dispersion[:,1:], 'b')
        for pt in self.path:
            ax.axvline(pt[1], color='k')
        ax.set_xticks([pt[1] for pt in self.path])
        ax.set_xticklabels([pt[0].label or '' for pt in self.path])
        if show:
            fig.show()
        return fig

    def as_dict(self):
        return jsanitize(super(PhononDispersion, self).as_dict(),strict=True)

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["dispersion"]), [(Qpoint.from_dict(q[0]), q[1]) for q in d["path"]])

class LTE(MSONable):
    """
    Contains the objects which can be parsed from the lte directory
    Args:
        disp_patterns:
            DispPatterns object
        freq_dos:
            FreqDos object
        energies:
            Energies object
        bzpoints:
            BZPoints object
        inputs:
            CaesarInputs object
        phonon_disp:
            PhononDispersion object
        dynmat:
            DynMat object
        lte_directory:
            location of lte directory
    """
    def __init__(self, lte_directory, disp_patterns, freq_dos, energies, bzpoints, inputs, phonon_disp, dynmat=None):
        self.disp_patterns = disp_patterns
        self.freq_dos = freq_dos
        self.energies = energies
        self.bzpoints = bzpoints
        self.inputs = inputs
        self.phonon_disp = phonon_disp
        self.dynmat = dynmat
        self.lte_directory = lte_directory

    @classmethod
    def from_files(cls, lte_dir, dynmat=False):
        """
        Args:
            lte_dir:
                location of caesar's lte directory

        Returns:
            LTE object
        """
        disp_patterns = DispPatterns.from_files(lte_dir)
        freq_dos = FreqDos.from_files(lte_dir)
        energies = Energies.from_files(lte_dir)
        bzpoints = BZPoints.from_files(lte_dir)
        inputs = CaesarInputs.from_files(lte_dir)
        phonon_disp = PhononDispersion.from_files(lte_dir)
        if dynmat:
            dynmat = DynMat.from_files(lte_dir)
        else:
            dynmat = None
        return cls(os.path.abspath(lte_dir), disp_patterns, freq_dos, energies, bzpoints, inputs, phonon_disp, dynmat)

    @classmethod
    def from_dict(cls, d):
        if d["dynmat"] is None:
            return cls(d["lte_directory"], DispPatterns.from_dict(d["disp_patterns"]), FreqDos.from_dict(d["freq_dos"]), Energies.from_dict(d["energies"]), BZPoints.from_dict(d["bzpoints"]), CaesarInputs.from_dict(d["inputs"]), PhononDispersion.from_dict(d["phonon_disp"]))
        else:
            return cls(d["lte_directory"], DispPatterns.from_dict(d["disp_patterns"]), FreqDos.from_dict(d["freq_dos"]), Energies.from_dict(d["energies"]), BZPoints.from_dict(d["bzpoints"]), CaesarInputs.from_dict(d["inputs"]), PhononDispersion.from_dict(d["phonon_disp"]), DynMat.from_dict(d["dynmat"]))

    def plot(self, show=True):
        """
        Plot the dispersion curves and dos together using matplotlib
        Args:
            show:
                whether or not to show the plot (default is True)
        Returns:
            a matplotlib figure object with a plot of the dispersion curve and dos
        """
        fig = plt.figure(figsize=(12,6))
        gs = plt.GridSpec(1,4)
        ax_disp = fig.add_subplot(gs[0,:3])
        ax_dos = fig.add_subplot(gs[0,3:], sharey = ax_disp)
        ax_disp.set_xlim(self.phonon_disp.dispersion[:,0].min(), self.phonon_disp.dispersion[:,0].max())
        ax_disp.set_xlabel('k-space')
        ax_disp.set_ylabel('Energy (eV)')
        ax_disp.plot(self.phonon_disp.dispersion[:,0], self.phonon_disp.dispersion[:,1:], 'b')
        for pt in self.phonon_disp.path:
            ax_disp.axvline(pt[1], color='k')
        ax_disp.set_xticks([pt[1] for pt in self.phonon_disp.path])
        ax_disp.set_xticklabels([pt[0].label or '' for pt in self.phonon_disp.path])
        ax_dos.plot(self.freq_dos.dos[:,1], self.freq_dos.dos[:,0])
        ax_dos.set_xlabel('Density of states')
        ax_dos.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax_dos.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        if show:
            fig.show()
        return fig

class CaesarDisplacement(MSONable):
    """
    Corresponds to a displacement of an atom preformed in the phonon calculation
    Args:
        amplitude:
            amplitude of the displacement
        disp:
            index of the atom and the displacement as recorded by caesar
        positive:
            doc of the entire vasp run for displacement in the positive direction (as a dictionary)
        negative:
            doc of the entire vasp run for displacement in the negative direction (as a dictionary)
    """
    def __init__(self, amplitude, disp, positive, negative):
        self._amplitude = amplitude
        self._disp = disp
        self._positive = positive
        self._negative = negative

    @classmethod
    def from_files(cls, disp_dir):
        """
        Args:
            disp_dir:
                directory corresponding to this displacement (usually written by caesar as atomn.dispm)

        Returns:
            CaesarDisplacement object
        """
        amplitude = float(np.genfromtxt(os.path.join(disp_dir, 'amplitude.dat')))
        disp = np.genfromtxt(os.path.join(disp_dir, 'disp.dat')).tolist()
        #drone = VaspToComputedEntryDrone(inc_structure=True)
        #TODO: Possibly either find another drone or combine Vasprun and Outcar here
        #       because this drone will return a dict, not an object. Don't forget to modify from_dict and the docstring if you make this change.
        drone = VaspToDbTaskDrone(simulate_mode=True)
        print 'assimilating vaspruns in ', disp_dir
        positive = drone.assimilate(os.path.join(disp_dir, 'positive'))
        #returned dict has last_updated as datetime.datetime object which jsanitize complains about
        if 'last_updated' in positive:
            positive['last_updated'] = str(positive['last_updated'])
        negative = drone.assimilate(os.path.join(disp_dir, 'negative'))
        if 'last_updated' in negative:
            negative['last_updated'] = str(negative['last_updated'])
        return cls(amplitude, disp, positive, negative)

    @classmethod
    def from_dict(cls, d):
        #return cls(d['amplitude'], d['disp'], ComputedStructureEntry.from_dict(d['positive']), ComputedStructureEntry.from_dict(d['negative']))
        return cls(d['amplitude'], d['disp'], d['positive'], d['negative'])

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def disp(self):
        return self._disp

    @property
    def positive(self):
        return self._positive

    @property
    def negative(self):
        return self._negative


class CaesarSupercell(MSONable, MutableSequence):
    """
    Represents information about all calculations preformed for a given supercell

    Args:
        primitive_structure:
            pymatgen structure representing primitive cell
        super_equilibrium:
            pymatgen structure representing supercell
        supercell:
            the matrix which transforms that lattice of primitive_structure to super_equilibrium
        super_spin:
            list of spins cooresponding to the atoms in super_equilibrium (leave empty for nonspin calc)
        displacements:
            list of CaesarDisplacement objects for displacements in this supercell
    """
    def __init__(self, primitive_structure, primitive_spin, super_equilibrium, supercell, super_spin=None, displacements=[]):
        self._primitive_structure = primitive_structure
        self._primitive_spin = primitive_spin
        self._super_equilibrium = super_equilibrium
        self._supercell = supercell
        self._super_spin = super_spin
        self._displacements = displacements

    @classmethod
    def from_files(cls, sc_dir, get_displacements=True):
        """
        Args:
            sc_dir:
                directory containing this supercell's calculations (usually written by caesar as Supercell_i)
            get_displacements:
                whether or not to also parse all displacement calculations preformed within this supercell (default is True)

        Returns:
            CaesarSupercell object
        """
        primitive_structure, primitive_masses, primitive_spin = readCaesarEquilibriumStructure(os.path.join(sc_dir, 'equilibrium.dat'), os.path.join(sc_dir, 'lattice.dat'))
        super_equilibrium, super_masses, super_spin = readCaesarEquilibriumStructure(os.path.join(sc_dir, 'super_equilibrium.dat'), os.path.join(sc_dir, 'super_lattice.dat'))
        supercell = np.genfromtxt(os.path.join(sc_dir, 'supercell.dat'))
        displacements = []
        if get_displacements:
            for disp in sorted(glob(os.path.join(sc_dir, 'atom*disp*'))):
                displacements.append(CaesarDisplacement.from_files(disp))
        return cls(primitive_structure, primitive_spin, super_equilibrium, supercell, super_spin, displacements)

    def as_dict(self):
        return jsanitize(super(CaesarSupercell, self).as_dict(),strict=True)

    @classmethod
    def from_dict(cls, d):
        return cls(pmg.Structure.from_dict(d['primitive_structure']), d['primitive_spin'], pmg.Structure.from_dict(d['super_equilibrium']), np.array(d['supercell']), d['super_spin'], [CaesarDisplacement.from_dict(disp) for disp in d['displacements']])

    @property
    def displacements(self):
        return self._displacements
    @property
    def primitive_spin(self):
        return self._primitive_spin

    @property
    def super_spin(self):
        return self._super_spin

    @property
    def supercell(self):
        return self._supercell

    @property
    def super_equilibrium(self):
        return self._super_equilibrium

    @property
    def primitive_structure(self):
        return self._primitive_structure

    def __getitem__(self, ind):
        return self.displacements[ind]

    def __len__(self):
        return len(self.displacements)

    def __delitem__(self, i):
        self._displacements.__delitem__(i)

    def insert(self, i, mode):
        if isinstance(mode, CaesarDisplacement):
            self._displacements.insert(i, mode)
        else:
            raise TypeError("Elements of CaesarSupercell must be CaesarDisplacement objects")

    def __setitem__(self, i, mode):
        if isinstance(mode, CaesarDisplacement):
            self._displacements[i] = mode
        else:
            raise TypeError("Elements of CaesarSupercell must be CaesarDisplacement objects")

class CaesarRun(MSONable):
    """
    Contains all information which can be parsed from a caesar run
    Args:
        lte:
            LTE object

        supercells: List of CaesarSupercell objects
    """
    def __init__(self, lte, supercells):
        self._lte = lte
        self._supercells = supercells

    @classmethod
    def from_files(cls, directory):
        """
        Args:
            directory:
                directory containing this caesar run

        Returns:
            CaesarRun object
        """
        lte = LTE.from_files(os.path.join(directory, 'lte'))
        supercells = [CaesarSupercell.from_files(sc) for sc in sorted(glob(os.path.join(directory, 'Supercell_*')))]
        return cls(lte, supercells)

    def as_dict(self):
        return jsanitize(super(CaesarRun, self).as_dict(),strict=True)

    @classmethod
    def from_dict(cls, d):
        return cls(LTE.from_dict(d['lte']), [CaesarSupercell.from_dict(sc) for sc in d['supercells']])

    @property
    def lte(self):
        return self._lte

    @property
    def supercells(self):
        return self._supercells

class CaesarToDBTaskDrone(AbstractDrone):
    """
    Drone for assimilating caesar runs in to database
    """
    def __init__(self, host="127.0.0.1", port=27017, database="vasp",
                 user=None, password=None, collection="tasks",
                 parse_dos=False, compress_dos=False, simulate_mode=False,
                 additional_fields=None, update_duplicates=True):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.collection = collection
        self.port = port
        self.simulate = simulate_mode
        if isinstance(parse_dos, six.string_types) and parse_dos != 'final':
            raise ValueError('Invalid value for parse_dos')
        self.parse_dos = parse_dos
        self.compress_dos = compress_dos
        self.additional_fields = additional_fields or {}
        self.update_duplicates = update_duplicates
        if not simulate_mode:
            conn = MongoClient(self.host, self.port, j=True)
            db = conn[self.database]
            if self.user:
                db.authenticate(self.user, self.password)
            if db.counter.find({"_id": "taskid"}).count() == 0:
                db.counter.insert({"_id": "taskid", "c": 1})

    def _insert_doc(self, d):
        #coppied mostly from VaspToDBTaskDrone except for handling of dos since vasp runs are not storred with key 'calcs'
        if not self.simulate:
            # Perform actual insertion into db. Because db connections cannot
            # be pickled, every insertion needs to create a new connection
            # to the db.
            conn = MongoClient(self.host, self.port)
            db = conn[self.database]
            if self.user:
                db.authenticate(self.user, self.password)
            coll = db[self.collection]

            # Insert dos data into gridfs and then remove it from the dict.
            # DOS data tends to be above the 4Mb limit for mongo docs. A ref
            # to the dos file is in the dos_fs_id.
            result = coll.find_one({"dir_name": d["dir_name"]},
                                   ["dir_name", "task_id"])
            if result is None or self.update_duplicates:
                if self.parse_dos:
                    for supercell in d['supercells']:
                        for disp in supercell['displacements']:
                            for calc in [disp['positive'], disp['negative']]:
                                if "dos" in calc:
                                    dos = json.dumps(calc["dos"], cls=MontyEncoder)
                                    if self.compress_dos:
                                        dos = zlib.compress(dos, self.compress_dos)
                                        calc["dos_compression"] = "zlib"
                                    fs = gridfs.GridFS(db, "dos_fs")
                                    dosid = fs.put(dos)
                                    calc["dos_fs_id"] = dosid
                                    del calc["dos"]
                d["last_updated"] = datetime.datetime.today()
                if result is None:
                    if ("task_id" not in d) or (not d["task_id"]):
                        d["task_id"] = db.counter.find_and_modify(
                            query={"_id": "taskid"},
                            update={"$inc": {"c": 1}}
                            )["c"]
                    logger.info("Inserting {} with taskid = {}"
                                .format(d["dir_name"], d["task_id"]))
                elif self.update_duplicates:
                    d["task_id"] = result["task_id"]
                    logger.info("Updating {} with taskid = {}"
                                .format(d["dir_name"], d["task_id"]))
                coll.update({"dir_name": d["dir_name"]}, {"$set": d},
                            upsert=True)
                return d["task_id"]
            else:
                logger.info("Skipping duplicate {}".format(d["dir_name"]))
        else:
            d["task_id"] = 0
            logger.info("Simulated insert into database for {} with task_id {}"
                        .format(d["dir_name"], d["task_id"]))
            return d


    def assimilate(self, path):
        """
        Parse entire caesar run and insert in to db

        Args:
            path:
                Location of caesar run

        Returns:
            task_id: task id of inserted run
        """
        run = CaesarRun.from_files(path)
        d = run.as_dict()
        fullpath = os.path.abspath(path)
        d["dir_name"] = fullpath
        state_successful = []
        for sc in run.supercells:
            for disp in sc.displacements:
                for c in [disp.positive, disp.negative]:
                    state_successful.append(c["state"]=='successful')
        if all(state_successful):
            d["state"] = 'successful'
        else:
            d["state"] = 'unsuccessful'

                    
        #assuming first supercell is gamma point
        d2 = run.supercells[0].displacements[0].positive
        for root_key in ["unit_cell_formula", "reduced_cell_formula", "pretty_formula",
                         "elements", "nelements", "is_hubbard", "hubbards", "run_type"]:
            d[root_key] = d2[root_key]
        for k,v in self.additional_fields.items():
            d[k] = v
        tid = self._insert_doc(d)
        return tid

    def get_valid_paths(self, path):
        """
        Args:
            path:
                input path as a tuple generated from os.walk, i.e.,
                (parent, subdirs, files).

        Returns:
            List of valid dir/file paths for assimilation
        """
        (parent, subdirs, files) = path
        if len(glob(os.path.join(parent, 'Supercell_*', 'atom*disp*', '*tive')))>0 and 'lte' in subdirs:
            return [parent]
        return []


def readCaesarEquilibriumStructure(equilibrium_dat, lattice_dat, get_spins=True):
    """
    Helper method for reading structures written by caesar
    Args:
        equilibrium_dat:
            dat file containing atomic positions
        lattice_dat:
            dat file containing lattice
        get_spins:
            whether or not to read spins from equilibrium_dat (default is True) 

    Returns:
        if get_spins==True:
            returns a tuple (structure, masses, spins) where structure is a pymatgen structure and masses, spins are numpy arrays
        if get_spins==False:
            returns a tuple (structure, masses) where structure is a pymatgen structure and masses is a numpy array
    """
    latdat = np.genfromtxt(lattice_dat)
    lattice = pmg.Lattice(BOHR_TO_ANG*latdat)
    with open(equilibrium_dat) as f:
        num_sites = int(f.readline())
        species = []
        coords = []
        spins = []
        masses = []
        for i in range(num_sites):
            line = f.readline().strip('\n').split()
            species.append(line[0])
            coords.append(line[2:5])
            if get_spins:
                spins.append(float(line[-1]))
            masses.append(float(line[1]))
        coords = np.array(coords, dtype=float)*BOHR_TO_ANG
    structure = pmg.Structure(lattice, species, coords, coords_are_cartesian=True)
    if get_spins:
        return (structure, masses, spins)
    else:
        return (structure, masses)

