import glob
import os
from typing import Generator, Optional
import numpy as np
import matplotlib.pyplot as plt

class DirectDetection:
    __VALID_MOLECULES = set() 
    __VALID_RANGE_TYPES = set() 
    __VALID_CONSTS = set() 
    __RANGE_TO_CONST = dict()
    __VALID_ENERGIES = set()

    __UNITS_TO_GeV = {
        "eV": 1e9,
        "keV": 1e6,
        "MeV": 1e3,
        "GeV": 1,
    }

    __inited = False

    AVOGADROS_CONSTANT = 6.0221415e23 # mol^-1
    ONE_MOLE_ANTHRACENE = AVOGADROS_CONSTANT * 24 # atoms
    DM_DENSITY = 0.4 # GeV cm^-3

    @staticmethod
    def init(directory:str = "data"):
        '''
        Initialises this python module for data validation.

        Keyword Arguments:
        directory:str -- the name of the directory where the data is stored.
        This directory should be of the structure e.g.:
        data
        |--> Ge_c4
           |--> data_c4_avDCS_GeSR_0_030GeV.txt
        |--> Xe_d4
        '''
        count = 0
        for d in glob.glob(f"{directory}/*"):
            if os.path.isdir(d):
                for f in glob.glob(f"{d}/*"):
                    cnst, mol, rt, en, uts = DirectDetection.parse_fname(f.split("/")[-1])
                    DirectDetection.__VALID_MOLECULES.add(mol)
                    DirectDetection.__VALID_RANGE_TYPES.add(rt)
                    DirectDetection.__VALID_CONSTS.add(cnst)
                    DirectDetection.__VALID_ENERGIES.add(DirectDetection.toGeV(en, uts))
                    DirectDetection.__RANGE_TO_CONST[rt] = cnst 
                    count += 1
                    print(f"\rFound: {count} files.",end="")
        print("..Done!")
        DirectDetection.__inited = True
    
    @staticmethod
    def is_init() -> bool:
        return DirectDetection.__inited

    @staticmethod
    def parse_fname(filename:str) -> tuple[str, str, str, float, str]:
        '''
        Extract data about the experiment from the filename of a data file.

        Keyword Arguments:
        filename:str -- the filename to be parsed

        Returns:
        tuple (constant:str, molecule:str, range type:str, energy:float, units:str)
        '''
        fname = os.path.splitext(os.path.basename(filename))[0]
        _, const, _, molecule, unit, decimal = fname.split("_")
        zcount = 2 if int(unit) >= 10 else 3
        molecule, range_type = molecule[:2], molecule[2:]
        decimal, units = decimal[:zcount], decimal[zcount:]
        energy = int(unit) + int(decimal)/(10 ** zcount)
        return (const, molecule, range_type, energy, units)

    @staticmethod
    def get_fname(molecule:str, range_type:str, energy:float):
        if not DirectDetection.is_init():
            raise Exception("Call DirectDetection.init(Optional[directory]) before any other methods.")
        const = DirectDetection.__RANGE_TO_CONST[range_type]
        unit = int(np.floor(energy))
        zcount = 2 if unit >= 10 else 3
        dec = str(int((energy - unit) * 10**zcount))
        return f"data/{molecule}_{const}/data_{const}_avDCS_{molecule}{range_type}_{unit}_{dec.zfill(zcount)}GeV.txt"

    @staticmethod
    def toGeV(energy:float, uts:str) -> float:
        '''
        Converts a float in some eV units uts to GeV.

        Keyword arguments:
        energy:float -- the energy to be converted
        uts:str -- the units energy is in. E.g. "keV"

        Returns:
        float : the energy converted to GeV
        '''
        return DirectDetection.__UNITS_TO_GeV[uts] * energy
    
    @staticmethod
    def is_valid_molecule(mol:str, rt:str, en:float) -> bool:
        if not DirectDetection.is_init():
            raise Exception("Call DirectDetection.init(Optional[directory]) before any other methods.")
        return mol in DirectDetection.__VALID_MOLECULES and rt in DirectDetection.__VALID_RANGE_TYPES and en in DirectDetection.__VALID_ENERGIES

    @staticmethod
    def get_valid_fnames(mol:Optional[str] = None, rt:Optional[str] = None, en:Optional[float] = None) -> Generator[str, None, None]:
        if not DirectDetection.is_init():
            raise Exception("Call DirectDetection.init(Optional[directory]) before any other methods.")
        for r in [rt] if rt else DirectDetection.__VALID_RANGE_TYPES:
            for m in [mol] if mol else DirectDetection.__VALID_MOLECULES:
                for e in [en] if en else DirectDetection.__VALID_ENERGIES:
                    yield DirectDetection.get_fname(m, r, e)

    class Plot:
        def __init__(self, ax, args, kwargs, command="plot"):
            self.ax = ax
            self.args = args
            self.kwargs = kwargs
            self.command = command
        def plot(self):
            getattr(self.ax, self.command)(*self.args, **self.kwargs)

        @staticmethod
        def set_sizes(w =20, h=20, fsz=30):
            plt.rc("figure", {"figsize": (w,h)})
            plt.rc("font", {"size": fsz})
    
    # in kwargs: scale
    @staticmethod
    def get_plot(x, y, **kwargs):
        args = []


    @staticmethod
    def get_csd_vs_T_plot(*args):
        """
        Returns a plot object of the cross-section differential against T

        May be called as:
        get_csd_vs_T_plot(filename:str)
        OR
        get_csd_vs_T_plot(mol:str, rt:str, en:float)
        """
        if len(args) == 1:
            filename = args[0]
        elif len(args) == 3:
            filename = DirectDetection.get_fname(*args)
        else:
            raise Exception("get_csd_vs_T_plot has two signatures: get_csd_vs_T_plot(filename:str) and get_csd_vs_T_plot(mol:str, rt:str, en:float)")

        ax = plt.gca()
        data = np.loadtxt(filename)
        const, mol, rtype, en, units = DirectDetection.parse_fname(filename)
        ax.set_xscale("log")
        ax.set_yscale("log")

        if mol == "Xe":
            ax.set_xlim([1e-2, 10])
            ax.set_ylim([1e-36, 1e-17])
        elif mol == "Ge":
            ax.set_xlim([10**(-1.5), 10])
            ax.set_ylim([1e-28, 1e-21])

        ax.set_xlabel(r"$T \text{(keV)}$") # energy transfer (in keV)
        ax.set_ylabel(r"$\frac{\text{d}\sigma\vec{v}}{\text{d}T} \text{cm}^3 \text{keV}^{-1} \text{day}^{-1}$") # averaged velocity-weighted differential cross sections (in cm^3/keV/day)
        margs = [data[:,0], data[:,1], "-"]
        kwargs = dict(linewidth=3, label=f"{mol} {rtype} {en} {units}")
        return DirectDetection.Plot(ax, margs, kwargs)

    @staticmethod
    def get_tmax(*args):
        if len(args) == 1:
            filename = args[0]
        elif len(args) == 3:
            filename = DirectDetection.get_fname(*args)
        else:
            raise Exception("get_plot has two signatures: get_plot(filename:str) and get_plot(mol:str, rt:str, en:float)")

        arr = np.loadtxt(filename)
        maxT = 0

        for x, y in arr:
            if y != 0.0:
                maxT = x
        _, _, _, en, _ = DirectDetection.parse_fname(filename)

        return maxT

    @staticmethod
    def get_tmax_matrix(Masses, Tmaxes, TmaxThresh):
        Tmaxes = np.array(Tmaxes, dtype=np.float64)
        Masses = np.array(Masses, dtype=np.float64)
        arr = np.vstack([Masses, Tmaxes]).T
        arr = arr[arr[:, 0].argsort()]
        arr = arr[arr[:,1] < TmaxThresh]
        return arr

    @staticmethod
    def fit_tmaxes(tmax_matrix):
        """
        Returns the gradient for a least squares fit on the tmax_matrix
        """
        A = np.vstack([tmax_matrix[:,0], np.zeros(len(tmax_matrix[:,0]))]).T
        return np.linalg.lstsq(A, tmax_matrix[:,1])[0][0]
    
    @staticmethod
    def get_dR_dT(m_x:float, dsv_dt, N_T:int = ONE_MOLE_ANTHRACENE, rho_x = DM_DENSITY):
        """
        Returns the array of computed dR/dT for each d(sigma v)/dt

        Keyword arguments:
        m_x:float -- mass of dark matter in GeV
        dsv_dt:np.array(float64) -- averaged velocity-weighted differential cross sections (in cm^3/keV/day)
        N_T:int -- number of atoms in detector (default: 1 mol of anthracene)
        rho_x:float -- density of dark matter in GeV cm^-3
        """
        return N_T * rho_x / m_x * dsv_dt


    @staticmethod
    def __cum_area(steps, values):
        total = 0
        totals = []
        totals.append(0)
        for i in range(len(steps)- 2, -1, -1):
            step_size = steps[i + 1] - steps[i]
            total += step_size * values[i]
            totals.append(total)
        return totals[::-1]



    
    @staticmethod
    def get_R_for_mx(m_x:float, T, dsv_dt, N_T:int = ONE_MOLE_ANTHRACENE, rho_x = DM_DENSITY):
        """
        Returns the array of computed R for a given m_x for each T, d(sigma v)/dt pair
        R[i] = integral from T[i] to Tmax of dR/dT

        Keyword arguments:
        m_x:float -- mass of dark matter in GeV
        T:np.array(float64) -- energy transfers
        dsv_dt:np.array(float64) -- averaged velocity-weighted differential cross sections (in cm^3/keV/day)
        N_T:int -- number of atoms in detector (default: 1 mol of anthracene)
        rho_x:float -- density of dark matter in GeV cm^-3
        """
        dR_dT = DirectDetection.get_dR_dT(m_x, dsv_dt, N_T, rho_x)
        return np.array(DirectDetection.__cum_area(T, dR_dT))

    # NOTE: Could be made more efficient
    @staticmethod
    def get_R_for_Tmin(Tmin:float, m_x, T, dsv_dt, N_T:int = ONE_MOLE_ANTHRACENE, rho_x = DM_DENSITY):
        """
        Returns the array of computed R for a given Tmin for each m_x, d(sigma v)/dt pair
        R[i] = integral from Tmin to Tmax of dR/dT

        Keyword arguments:
        Tmin:float -- energy transfer to look at
        m_x:np.array(float64) -- masses of dark matter in GeV
        T:np.array(float64) -- energy transfers
        dsv_dt:np.array(float64) -- averaged velocity-weighted differential cross sections (in cm^3/keV/day)
        N_T:int -- number of atoms in detector (default: 1 mol of anthracene)
        rho_x:float -- density of dark matter in GeV cm^-3
        """
        dR_dT = DirectDetection.get_dR_dT(m_x, dsv_dt, N_T, rho_x)
        Rs = []
        for m in m_x:
            R = DirectDetection.get_R_for_mx(m_x, T, dsv_dt, N_T, rho_x)
            Rs.append(R[np.where(T == Tmin)])
        return np.array(Rs)
