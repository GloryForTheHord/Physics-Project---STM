import numpy as np
from ProjectModule import *
from lagrossebertha.plotting import plot as custom_plot
from lagrossebertha.schrodinger import mat_trans
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def Q2(N :int, d :float, V0 :float, Vapp :float):
    print("############################## Q2 ##############################")
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    xTab, vTab = trapezoid_rect(N, d, V0, Vapp)
    
    print(xTab, len(xTab), vTab, len(vTab))
    
    xCoord = [0, 0, d, d]
    yCoord = [0, V0+Vapp, V0, 0]
    
    trapezoid = patches.Polygon(list(zip(xCoord, yCoord)), linewidth=1.5, label="Trapèze", fill=False, color="blue")
    
    ax.add_patch(trapezoid)
    
    xCoord = []
    yCoord = []
    
    for i in range(N+1):
        xCoord.append(xTab[i])
        xCoord.append(xTab[i])
        yCoord.append(vTab[i])
        yCoord.append(vTab[i+1])        
    
    approx = patches.Polygon(list(zip(xCoord, yCoord)), linewidth=1.5, label="Approximation", fill=False, color="orange")
    
    ax.add_patch(approx)
    
    plt.legend(loc="best")
    plt.show()


def Q3(Z :np.ndarray, V0 :np.ndarray) -> None:
    print("############################## Q3 ##############################")

    plt.figure()

    for i in range(len(Z)):
        for v in V0:
            
            xtab = [0, Z[i]]
            Vtab = [0, v, 0]
            
            E = np.linspace(1/1000, v-1/1000, 1000)

            T = mat_trans(xtab, np.array(Vtab, dtype=np.double), np.array(E, dtype=np.double))
            
            T2 = 100*np.abs(T)**2, "", "$V_0$ = {} eV".format(v)
            
            custom_plot(E, T2, xlabel="$E$ [eV]", ylabel="$T(E)$ [%]")
        plt.show()
        
    for z in Z:
        xtab = [0, z]
        Vtab = [0, V0[2], 0]
        
        E = np.linspace(1/1000, V0[2]-1/1000, 1000)

        T = mat_trans(xtab, np.array(Vtab, dtype=np.double), np.array(E, dtype=np.double))
        
        T2 = 100*np.abs(T)**2, "", "$z$ = {} nm".format(z*1e+9)
        
        custom_plot(E, T2, xlabel="$E$ [eV]", ylabel="$T(E)$ [%]")
    plt.show()


def Q4(d :float, V0 :float, Vapp :float, nmax :int) -> int:
    print("############################## Q4 ##############################")

    E = np.linspace(1/1000, V0-1/1000, 100)

    t, e, f, nopt = get_n_opt(lambda n: trapezoid_rect(n, d, V0, Vapp), E, nmax, func=True)

    plt.figure()

    custom_plot(np.array(range(1, nmax+1)), (t*1e3,"",""), xlabel="Nombre de découpes", ylabel="Temps d'exécution [ms]")
    plt.show()

    custom_plot(np.array(range(1, nmax+1)), (e*1e3,"","") , xlabel="Nombre de découpes", ylabel="Erreur relative $\\times 10^3$")
    plt.show()

    custom_plot(np.array(range(1, nmax+1)), (f,"",""), xlabel="Nombre de découpes", ylabel="Fonction d'optimisation")
    plt.plot(nopt, f[nopt-1], 'or', label="Optimum")
    plt.legend(loc="best")
    plt.show()

    return nopt #type: ignore @IDE_COMMAND


def Q5(Ssigma :float, V0 :float, Vapp :np.ndarray, z :np.ndarray, nopt :int =12) -> None:
    print("############################## Q5 ##############################")
    
    plt.figure()
    for zz in z:
        I = []
        for v in Vapp:
            I.append(current_1D(zz, v, V0, Ssigma, nopt))
        
        custom_plot(Vapp, (np.array(I)*1e3, '', r'$d = {}$ nm'.format(zz*1e9)))
    custom_plot(Vapp, xlabel="Différence de potentiel appliquée [V]", ylabel="Intensité du courant électrique [mA]")
    plt.show()
    
    
def Q6(Ssigma :float, V0 :float, Vapp :float, z0 :float, R0 :float, xmax :float, nopt :int =12) -> None:
    print("############################## Q6 ##############################")
    
    def z_step(x):
        if x < xmax/2:
            return z0
        else:
            return z0+0.25e-9
        
    I_1D_2D_curved_domain(Ssigma, V0, Vapp, z_step, R0, xmax, z0, nopt)
    
    
def Q_BONUS_sine(Ssigma :float, V0 :float, Vapp :float, z0 :float, R0 :float, xmax :float, amplitude :float, pulsation :float, nopt :int =12):
    print("############################## Q_SINE ##############################")
    
    def z_sine(x):
        return z0 + amplitude*np.sin(pulsation*x)

    I_1D_2D_curved_domain(Ssigma, V0, Vapp, z_sine, R0, xmax, z0, nopt)
    

def Q_BONUS_blocs(Ssigma :float, V0 :float, Vapp :float, z0 :float, R0 :float, xmax :float, amplitude :float, pulsation :float, nopt :int =12):
    print("############################## Q_SINE ##############################")
    
    def z_blocs(x):
        return z0 + amplitude*(((pulsation*x)//1)%2)

    I_1D_2D_curved_domain(Ssigma, V0, Vapp, z_blocs, R0, xmax, z0, nopt)
    

if __name__ == "__main__":
    Q2(12, 0.5e-9, 5, 2)