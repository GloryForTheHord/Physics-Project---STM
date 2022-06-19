
############################## IMPORTS ##############################


import numpy as np
from packages.schrodinger import *
from packages.plotting import plot as custom_plot
import time
from alive_progress import alive_bar    #* Since Q6 take a consequent time to execute, a progress bar is provided



############################## Q2 ##############################


def trapezoid_rect(N, d, V0, Vapp):
    V_tab = [0]
    X_tab = []
    
    for i in range(N):
        V_tab.append(V0+(N-i)*Vapp/N)
        X_tab.append(i*d/N)

    X_tab.append(d)
    V_tab.append(0)
    
    return X_tab, V_tab


############################## Q4 ##############################


def get_n_opt(slicer, E, nmax, func=False):
    
    xtab, Vtab = slicer(nmax)
    
    transNMax = [mat_trans(xtab, np.array(Vtab, dtype=np.double), np.array([e], dtype=np.double)) for e in E]
    
    N = len(E)
    epsilonN = np.array([], dtype=np.double)
    epsilonTemp = np.zeros(N)
    timeN    = np.array([], dtype=np.double)
    timeTemp = np.zeros(N)
    
    for n in range(1, nmax+1):
        for i in range(N):
            
            timeTemp[i] = time.time()
        
            xtab, Vtab = slicer(n)
            transTemp = mat_trans(xtab, np.array(Vtab, dtype=np.double), np.array([E[i]], dtype=np.double))
        
            timeTemp[i] = time.time() - timeTemp[i]
            
            epsilonTemp[i] = np.abs(transTemp)**2 - np.abs(transNMax[i])**2
            
        timeN = np.append(timeN, np.mean(timeTemp))
        epsilonN = np.append(epsilonN,  1/N * np.sqrt(np.sum(np.square(epsilonTemp))))
        
    epsilon1 = epsilonN[0]
    timeNMax = timeN[-1]
    
    fn = [epsilonN[n]/epsilon1 + timeN[n]/timeNMax for n in range(0, nmax)]
    
    if func:
        return timeN, epsilonN, fn, np.argmin(fn)+1
    
    return np.argmin(fn)+1


############################## Q5 ##############################

def current_1D(z, Vapp, V0, Ssigma, nopt):
    xtab, Vtab = trapezoid_rect(nopt, z, V0, Vapp)
    E = np.linspace(1/10000, Vapp-1/10000, 100)
    T2 = np.abs(mat_trans(xtab, np.array(Vtab, dtype=np.double), E))**2
    return Ssigma * np.sum(T2) * Vapp/100


############################## Q6 ##############################

def current_2D_unscaled_disk(x0, z, R0, Vapp, V0, Ssigma, nopt):
    x = np.linspace(-R0, R0, 100)
    zt = [z(x0+xx) + R0 - np.sqrt(R0**2 - xx**2) for xx in x]
    
    I = [current_1D(zz, Vapp, V0, Ssigma, nopt) for zz in zt]
    
    I2D = np.sum(I)*2*R0/100
    
    return I2D


def get_S(Ssigma :float, V0 :float, Vapp :float, z0 :float, R0 :float, nopt :int =12):
    def z_flat(x):
        return z0
    
    I1D = current_1D(z_flat(0), Vapp, V0, Ssigma, nopt)
    
    I2D = current_2D_unscaled_disk(0, z_flat, R0, Vapp, V0, Ssigma, nopt)
    
    return (I2D/I1D)**2


def current_1D_2D_curved_domain(Ssigma, V0, Vapp, z :Callable, R0, xmax, z0, nopt) -> None:
    
    S = get_S(Ssigma, V0, Vapp, z0, R0, nopt)
    
    print("S =", S, "=> sqrt(S) =", np.sqrt(S))    
    
    xDom1 = np.linspace(0, xmax, 500)
    xDom2 = np.linspace(0, xmax, 500)
    zDom1 = [z(x) for x in xDom1]
    
    I1D = I2D = []
    
    I1D = [current_1D(zz, Vapp, V0, Ssigma, nopt) for zz in zDom1]
    
    with alive_bar(len(xDom2), title="Progression de la pointe circulaire") as update_bar:
        for x in xDom2:
            update_bar()
            I2D.append(current_2D_unscaled_disk(x, z, R0, Vapp, V0, Ssigma, nopt))
        
    I2D = 1/np.sqrt(S) * np.array(I2D)
        
    custom_plot(np.array(xDom1)*1e9, (np.array(I1D)*1e6, "", "Pointe parfaite"), xlabel=r"Position [nm]", ylabel= r"Intensité du courant électrique [$\mu$A]")
    custom_plot(np.array(xDom2)*1e9, (np.array(I2D)*1e6, "", "Profil circulaire"))
    plt.show()