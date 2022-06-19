############################## IMPORTS ##############################


from typing import Callable
from scipy.sparse.linalg import eigs
from scipy.sparse import diags
from findiff import FinDiff
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema



############################## CONSTANTS ##############################


EV      = 1.6022e-19
M       = 9.11e-31
HBAR    = 6.626e-34/(2*np.pi)



############################## RESOLUTION ##############################


def fdm_states(V :Callable[[float], float], xlim :tuple, Elim :tuple, h :int = 1000, nbSol :int = 10) -> tuple[list[float], np.ndarray]:
    """
    Description:
    ------------
    
    Solves 1D Schrodinger equation at steady-state for a particle in a box

    Parameters:
    -----------
        `V`: Callable[[float], float]
            Value of the potential V with respect to the position x
        `xlim`: tuple
            Limits of the spatial domain
        `Elim`: tuple
            Limits of the accepted energy levels
        `h`: int, optional
            Precision of the method. Defaults to 1000.
        `nbSol`: int, optional
            Maximum  number of energy levels accepted. Defaults to 10.

    Returns:
    --------
        `E, psi`: tuple[list[float], np.ndarray[float, float]]
            `E` is a list containing the energy levels accepted by the domain. `psi` is a 2D array contaning on each column the wave function for the corresponding E.
    """
    
    x = np.linspace(xlim[0], xlim[1], h)
    Vtab = [V(xx)*EV for xx in x]
    
    e, p = eigs(-HBAR**2/(2*M) * FinDiff(0, x[1]-x[0], 2).matrix(x.shape) + diags(Vtab), k=nbSol, which='SR')
    
    indexes = [i for i in range(len(e)) if (Elim[0] < e[i]/EV < Elim[1])]
    
    e = e[indexes].real/EV
    
    sortIndexes = np.argsort(e)
    
    E = [e[i] for i in sortIndexes]
    psi = np.array([p[:, i].real for i in sortIndexes]).T
    
    return E, psi


def mat_states_semi_infinite(a, V0, hX=1000, hE=1000):
    
    V0 *= EV
    
    cstar = []
    
    Et = np.linspace(-V0, -0.001*EV, hE)
    residues = np.zeros_like(Et)
    
    for i in range(len(Et)):
    
        k1 = np.sqrt(-2*M/HBAR**2 *(Et[i] + V0) + 0*1j)
        k2 = np.sqrt(-2*M/HBAR**2 * Et[i]       + 0*1j)
        
        Mat = M_cont(k1, k2, a)
        
        residues[i] = abs(Mat[1, 0] - Mat[1, 1])
        
    indexes = argrelextrema(residues, np.less)
    
    E = Et[indexes]
    x = np.linspace(0, 2*a, hX)
    
    psi = np.zeros((len(E), len(x)), dtype=complex)
    
    def Psi(cstar, x, e):
        if x < 0 : return 0
        elif x < a :
            k1 = np.sqrt(-2*M/HBAR**2 *(e + V0) + 0*1j)
            return np.exp(-k1*x) - np.exp(k1*x)
        else :
            k2 = np.sqrt(-2*M/HBAR**2 * e       + 0*1j)
            return cstar*np.exp(-k2*x)
    
    for i in range(len(E)):
        k1 = np.sqrt(-2*M/HBAR**2 *(E[i] + V0) + 0*1j)
        k2 = np.sqrt(-2*M/HBAR**2 * E[i]       + 0*1j)
        
        Mat = M_cont(k1, k2, a)
        
        cstar = abs(Mat[0, 0] - Mat[0, 1])
        
        psi[i] = [Psi(cstar, xx, E[i]) for xx in x]
        
    return E/EV, psi.T

def mat_trans(xtab :list, Vtab :np.ndarray, E :np.ndarray):
        
    E[:] = E[:]*EV
    Vtab[:] = Vtab[:] * EV
    T = np.array([], dtype=complex)
    
    for e in E:
        
        McontTot = np.eye(2,dtype=complex)
        
        for i in range(len(xtab)):
            
            k1 = np.sqrt(-2*M/HBAR**2 *(e - Vtab[i])   + 0*1j)
            k2 = np.sqrt(-2*M/HBAR**2 *(e - Vtab[i+1]) + 0*1j)
            
            McontTot = np.matmul(M_cont(k1, k2, xtab[i]), McontTot)
       
        T = np.append(T, McontTot[0, 0] - McontTot[0, 1]*McontTot[1, 0]/McontTot[1, 1])
        
    return T


############################## FUNCTIONS ##############################

    
def M_cont(k1, k2, x):
    
    M1 = np.array([[np.exp(-k1*x), np.exp(k1*x)],[-k1*np.exp(-k1*x), k1*np.exp(k1*x)]])
    M2 = np.array([[np.exp(-k2*x), np.exp(k2*x)],[-k2*np.exp(-k2*x), k2*np.exp(k2*x)]])
    
    return np.matmul(np.linalg.inv(M2),M1)


def show(xlim :tuple, ylim :tuple, V, E :list, psi :np.ndarray):
    
    x = np.linspace(xlim[0], xlim[1], len(psi))
    Vtab = np.array([V(xx) for xx in x])
    
    scale = 0.7*(E[1] - E[0])
    
    plt.figure()
    
    plt.ylim(ylim)
    
    plt.plot(x, Vtab, 'k')
    
    for i in range(len(E)):
        j = len(E) - i
        
        psi2 = abs(psi[:,j-1])**2
        norm = psi2[argrelextrema(psi2, np.greater)[0][0]]

        plt.plot(x, E[j-1] + scale*psi2/norm, label = "E[" + str(i) + "] = " + str(round(E[j-1], 2)) + r" $eV$")
        
    plt.legend(loc="best")
    
    plt.show()