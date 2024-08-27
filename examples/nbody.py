### A simple N+body + stellar evolution calculations. In this example, only the mass
### of the star is affected by the AGN disk, and it does not experience any change
### in velocity due to mass loss or accretion. To account for these effects, one may
### want to make use of the 'getExtras' function, to access the accretion rate onto
### and mass loss rate from the star. The same function would also provide the stellar
### luminosity, which might be useful for calculating feedback of the star on the disk,
### which is not considered here. 
from starsam import sam
import numpy as np
import matplotlib
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot as plt
plt.figure(figsize=(7.1,3.05))

### The N-body part of the calculation uses G=AU=Msun units, so 1 year is 2*pi
mbh = 1.0*10**7	# SMBH mass in msun
vfactor = 75499 # Nbody velocity units to CGS

cs0 = 10**5	# sound speed in CGS
X0 = 0.70	# ambient abundances
Y0 = 0.26
Z0 = 1.0 - X0 - Y0

rho0 = 10**-14	# density normalization
r0 = 10		# radius normalization
def diskrho(r):	# disk density profile
  rho = rho0*(r/r0)**-0.5
  return rho

def diskv(r):	# disk velocity profile
  return np.sqrt(mbh/r)

#The main ODE update. Includes both N-body and stellar parts
def Fdot(f):
  fs = f[:3]		#stellar mass parameters
  p1 = f[3:9]		#stellar position, velocity
  p2 = f[9:15]		#SMBH position, velocity
  ms = np.sum(fs)

  dx = p2[:3] - p1[:3]
  dr = np.sum(dx**2)**0.5
  fg = dx*mbh/dr**3

  update = np.zeros(15)

  update[3:6]  = p1[3:6]
  update[9:12] = p2[3:6]

  update[6:9] = fg
  update[12:15] = -fg*ms/mbh

  #stellar stuff
  disk_vel = diskv(dr)
  dv = p1[3:6] - p2[3:6]
  v_rel = np.empty(3)
  v_rel[0] = dv[0] - disk_vel*dx[0]/dr
  v_rel[1] = dv[1] - disk_vel*dx[1]/dr
  v_rel[2] = dv[2]
  vrel = vfactor*np.sqrt(np.sum(v_rel**2))

  dfs = sam.fdot(0.0, fs, diskrho(dr), cs0, X0, Y0, Z0, v0=vrel, mdot_method="bhl")
  dfs *=0.5/np.pi
  update[:3] = dfs	# assumes that accretion has a negligible effect on the stellar velocity

  return update


# RK4 update
def rk4up(f, dt):
  k1 = Fdot(f)
  k2 = Fdot(f + k1*dt/2)
  k3 = Fdot(f + k2*dt/2)
  k4 = Fdot(f + k3*dt)
  return f + dt*(k1 + 2*k2 + 2*k3 + k4)/6

#Initialize an e=0.4, q=10**-6 binary with a period of ~10^6 years
f0 = np.zeros(15)
f0[0] = 7.0 # Hydrogen mass in msun
f0[1] = 2.0 # Helium mass in msun
f0[2] = 1.0 # Metal mass in msun
f0[3] = 1291798.7082012917 # stellar x position
f0[7] = 3.2920480131329444 # stellary y velocity
f0[9] = -1.2917987082012918 # SMBH x position
f0[13] = -3.2920480131329448e-06 # SMBH y velocity

#en = [energy(f0)]
ms = [10]
dt = 1000.0
NT = int(10**7/dt)
for i in range(NT):
  f0 = rk4up(f0, dt)
  ms.append(np.sum(f0[:3]))

plt.plot(np.linspace(0, 1, NT), ms[:-1])

plt.xlim([0.0, 1])
plt.ylabel(r'$M_*\,(M_\odot)$', labelpad=-0.025)
plt.xlabel(r'$t\,(10^7{\rm\,yr})$', labelpad=-6.2)
plt.subplots_adjust(top=0.985, bottom=0.10, left=0.07, right=0.985)
plt.savefig('nbody_results.png', dpi=300)
