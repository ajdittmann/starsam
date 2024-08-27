## A script illustrating how to run calculations using a time dependent function for the ambient density ##
import numpy as np
from starsam import sam
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
plt.figure(figsize=(7.1,3.05))
cmap = plt.get_cmap('viridis_r')

#pick in intial mass and mass fractions for the star
M0 = 10.0
Xs = 0.74
Ys = 0.24
Zs = 1.0 - Xs - Ys

# AGN background abundances
X0 = 0.72
Y0 = 0.27
Z0 = 1.0 - X0 - Y0

# AGN background conditions
cs0 = 10**6

# Approximation for accretion rate onto star
mdot_method = "tidal"

# Simulation duration, result sampling cadence
Tend = 4*10**7
Nout = 1000
t_eval = 10**np.linspace(4, np.log10(Tend), Nout)
t_eval[-1] = Tend


rhos = 10.0**np.linspace(-19, -15.3, 11)
omega0 = 10.0**-10
for i in range(len(rhos)):
    rho0 = rhos[i]

    def my_rho(t):
        Dt = 5*10**6
        t0 = 10**7
        return rho0*0.5*(1.0 - np.tanh(  (t - t0)/Dt ) )

    t, m, msg =  sam.run(M0, Xs, Ys, Zs, X0, Y0, Z0, Tend, my_rho, cs0, mdot_method=mdot_method, omega0=omega0, full_output = False, t_eval = t_eval)

    if len(m)>1:
        rhostr = np.format_float_scientific(rho0, precision=2, exp_digits=2, unique=False)
        rhobase = float(rhostr[:4])
        rhoexp = int(rhostr[-2:])

        plt.plot(t, m, color=cmap(i/10), label=r"$\rho\!=\!%.2f\!\times\! 10^{-%d}\,{\rm g\,cm^{-3}}$" % (rhobase, rhoexp))

plt.yscale('log')
plt.xscale('log')
plt.xlim([10**4, 2.5*10**7])
plt.legend(frameon=False,loc=2, fontsize=9.5, borderpad=0.0, borderaxespad=0.2, labelspacing=0.3)

plt.ylabel(r'$M_*\,(M_\odot)$',labelpad=-1)
plt.xlabel(r'$t\,({\rm yr})$', labelpad=-6.2)
plt.subplots_adjust(top=0.985, bottom=0.10, left=0.06, right=0.985)
#plt.savefig('tdep_results.png', dpi=300)
plt.show()




