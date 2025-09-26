## A basic script illustrating how to run calculations over a grid of parameters, in this case the ambient density ##
import numpy as np
from starsam import sam
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis_r')

Nrho = 11

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
#mdot_method = "Bondi"
mdot_method = "tidal"

# Simulation duration, result sampling cadence
#Tend = 4*10**6
Tend = 6*10**6
Nout = 4000
t_eval = 10**np.linspace(0.1, np.log10(Tend), Nout)
t_eval[-1] = Tend

solver = 'RADAU'
#solver = 'RK45'
#solver = 'BDF'
#solver = 'DOP853'

rhos = 10.0**np.linspace(-18, -8.0, Nrho)
#rhos = 10.0**np.linspace(-18, -14.0, Nrho)
omega0 = 10.0**-6


for i in range(len(rhos)):
    rho0 = rhos[i]
    t, m, msg =  sam.run(M0, Xs, Ys, Zs, X0, Y0, Z0, Tend, rho0, cs0, tau0=10000.0, mdot_method=mdot_method, omega0=omega0, full_output = False, t_eval = t_eval, method=solver, rtol=10**-8, esc_reduce=True)
    print('bondi', rho0, msg)
    #if msg != "solve_ivp error":
    try: 
      rhostr = np.format_float_scientific(rho0, precision=2, exp_digits=2, unique=False)
      rhobase = float(rhostr[:4])
      rhoexp = int(rhostr[-2:])
      plt.plot(t, m, color=cmap(i/Nrho), label=r"$\rho\!=\!%.2f\!\times\! 10^{-%d}\,{\rm g\,cm^{-3}}$" % (rhobase, rhoexp))
    except ValueError:
      print('failure at t=0')


for i in range(len(rhos)):
    rho0 = rhos[i]
    t, m, msg =  sam.run(M0, Xs, Ys, Zs, X0, Y0, Z0, Tend, rho0, cs0, tau0=10000.0, mdot_method=mdot_method, omega0=omega0, full_output = False, t_eval = t_eval, method=solver, rtol=10**-12, esc_reduce=True, do_feedback=True)
    print('bondi', rho0, msg)
    #if msg != "solve_ivp error":
    try: 
      rhostr = np.format_float_scientific(rho0, precision=2, exp_digits=2, unique=False)
      rhobase = float(rhostr[:4])
      rhoexp = int(rhostr[-2:])
      plt.plot(t, m, color=cmap(i/Nrho),ls ="--")
    except ValueError:
      print('failure at t=0')


plt.yscale('log')
plt.xscale('log')
#plt.xlim([10**4, 2.5*10**7])
plt.legend(frameon=False,loc=2, fontsize=9.5, borderpad=0.0, borderaxespad=0.2, labelspacing=0.3)
#plt.legend()

plt.ylabel(r'$M_*\,(M_\odot)$',labelpad=-1)
plt.xlabel(r'$t\,({\rm yr})$', labelpad=-6.2)
plt.subplots_adjust(top=0.985, bottom=0.10, left=0.06, right=0.985)
#plt.savefig('base_results.png', dpi=300)
plt.show()

