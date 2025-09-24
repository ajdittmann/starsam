# starsam
Semi-analytical models of stellar evolution in AGN disks. See [the paper](https://arxiv.org/abs/2409.02981) for details. 

## Installation
First, clone the repository using ``git clone https://github.com/ajdittmann/starsam.git``.  
Then, install the package using ``pip install -e .`` or ``python setup.py install``. 

Although it is not required, if [numba](https://numba.pydata.org/) is installed it will be used to accelerate calculations.

## Usage
After installation, you can simulate the evolution of stars in dense environments. These solve the ODEs approximating """main sequence""" (hydrogen-burning) stellar evolution in AGN disks using scipy's ``solve_ivp`` along with vetting the runtime options and determining which of the multiple stopping criteria was employed. 

To use this in a script, after importing ``starsam``, simply use the ``sam.run`` function. 
```
import starsam

t, m, outcome = starsam.sam.run(Ms, Xs, Ys, Zs, X0, Y0, Z0, Tend, ...)
```
which will output arrays for some arbitrary times (in years), the stellar mass (in solar masses) at each time, and how the integration terminated (runaway accretion, hydrogen exhaustion, or time limitations).
In the above example, ``Ms`` is the initial stellar mass (in solar masses); ``Xs``, ``Ys``, and ``Zs`` are the initial stellar mass fractions; ``X0``, ``Y0``, and ``Z0`` are the mass fractions of the AGN disks, which could be scalars or functions of time (e.g., ``X0(t)`` returns the value of ``X0`` at time ``t`` (in years)); and ``Tend`` is the final time of the simulation (in years, which must be greater than zero).   

The full call signature of ``sam.run`` is 
```
sam.run(Ms, Xs, Ys, Zs, X0, Y0, Z0, Tend, rho0=10**-18, cs0=10**6, v0=None, omega0=None, h0=None, Mbh=None, alpha=None, mdot_method="bondi", full_output=False, t_eval=None, method='RK45', rtol=None, atol=None, tkh=None, fnu=0.1)
```
Here, the optional arguments are
* ``rho0``, the density (in cgs) of the AGN disk at the location of the star. Functions of time are allowed. Used in every accretion prescription.
* ``cs0``, the sound speed (in cgs) of the AGN disk at the location of the star. Functions of time are allowed. Used in every accretion prescription.
* ``v0``, the relative velocity (in cgs) between the star and AGN disk at the location of the star. Functions of time are allowed. Used for the ``bhl`` and ``smh`` accretion prescriptions.
* ``omega0``, the angular velocity (in cgs) of the AGN disk at the location of the star. Functions of time are allowed. Used in ``tidal``, ``smh``, and ``gap`` accretion prescriptions.
* ``h0``, the aspect ratio (H/r) of the AGN disk at the location of the star. Functions of time are allowed. Used in the ``gap`` and ``smh`` accretion prescriptions.
* ``Mbh``, the mass of the central SMBH in solar masses. Used in the ``gap`` and ``smh`` accretion prescription.
* ``alpha``, the classic alpha viscosity parameter. Used in the ``gap`` accretion prescription.
* ``mdot_method``: valid options are the strings ``"bondi"``, ``"bhl"``, ``"smh"``, ``"tidal"``, and ``"gap"``.
* ``full_output ``: if ``False``, only returns time (yr), stellar mass (solar masses) and the termination condition as outputs. If ``True``, also outputs the mass of hydrogen within the star (solar masses), the mass of helium within the star (solar masses), the mass of metals within the star (solar masses), the mass accreted by the star over time (solar masses/yr), the mass lost by the star over time (solar masses/yr), the rate of hydrogen being fused into helium (solar masses /yr), the stellar luminosity (in solar luminosities), the stellar radius (in solar radii), and the stellar core temperature (in keV).
* ``t_eval``, an arry of times at which to output the stellar mass and other information if desired. Defaults to 1000 logarithmically spaced values between 10000 years and ``Tend``, or ``Tend/10`` if ``Tend<1000``.
* ``method``, the ODE algorithm used by ``solve_ivp``. Defaults to ``RK45``.
* ``rtol``, the relative error tolerance used by ``solve_ivp``. Defaults to ``1e-6``.
* ``atol``, the absolute error tolerance used by ``solve_ivp``. Defaults to ``rtol/1000`` so that ``rtol`` should dominate in most cases.
* ``tkh``, The stellar Kelvin-Helmholtz timescale in years, used to estimate when the star accretes faster than in can thermally adjust, leading to runaway accretion. May be a constant, or function of stellar mass, radius, and luminosity (in cgs). Defaults to the estimate for an n=3 polytrope.
* ``fnu``, The fraction of energy released via neutrinos during hydrogen fusion, by default 10%. 

## Advanced Usage
At its core, starsam solves a set of ordinary differential equations for the hydrogen, helium, and metal mass of a star in an AGN disk. The ``sam.run`` function solves these equations, given some initial conditions, using``scipy.integrate.solve_ivp``. If you would like to use your own integration method, or couple this model of stellar evolution to a larger system of differental equations (such as an N-body system), starsam also provides a ``sam.fdot`` function, which returns ``df/dt`` in solar masses/year, where ``f`` is an array of the stellar mass in hydrogen, helium, and metals. 

The full call signature of ``sam.fdot`` is 
```
sam.fdot(t, f, rho0, cs0, X0, Y0, Z0, v0=None, omega0=None, Mbh=None, h0=None, alpha=None, mdot_method="bondi", tkh=None, fnu=0.1):
```
Where ``t`` is the simulation time in years, ``f`` is an array containting the mass of the star in hydrogen, helium, and metals. The other variables have the same roles as described above in the ``sam.run`` function call.

starsam also proviedes a helper function in case you are using a different ODE solver and would like to know things like the stellar radius, mass loss rate, etc. This function is ``sam.getExtras``, which has the same call signature as ``sam.fdot``.

``sam.getExtras`` returns the accretion rate (solar masses/yr), the mass loss rate (solar masses/yr), hydrogen burning rate (solar masses /yr), the stellar luminosity (in solar luminosities), the stellar radius (in solar radii), and the stellar core temperature (in keV).

## Examples
A few examples demonstrating how to run some basic simulations can be found in the [examples](examples/) directory.

## References
This package solves a set of equations described in [Dittmann & Cantiello 2025](https://arxiv.org/abs/2409.02981). Essentially, it combines approximate models for aspects of stellar evolution in AGN disks (such as accretion and mass loss), primarily developed in these papers ([1](https://ui.adsabs.harvard.edu/abs/2021ApJ...910...94C/abstract),[2](https://ui.adsabs.harvard.edu/abs/2021ApJ...916...48D/abstract)), with approximate models of stellar structure developed in ([3](https://ui.adsabs.harvard.edu/abs/1964ApJS....9..201F/abstract),[4](https://ui.adsabs.harvard.edu/abs/1984ApJ...280..825B/abstract))
