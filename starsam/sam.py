import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp as ivp
# N.B. In this script, BAC refers to Bond, Arnett, and Carr 1984, ApJ, 280, 825

# Physical constants (cgs)
G = 6.6743*10**-8
msun = 1.989*10**33
mh = 1.6726*10**-24
rsun = 6.955*10**10
lsun = 3.839 *10**33
sbc = 1.8914785*10**-15  #Stefan-Boltzmann divided by c
kboltz = 1.380658*10**-16


# unit conversion factors
mev2erg = 1.60218*10**-6	# erg per MeV
spy = 3.154*10**7		# seconds per year

# The fraction (by number) of C, N, O atoms using Asplund, Amarsi, and Grevesse 2021
cnofrac_solar = 10.0**(8.46-12) + 10.0**(7.83-12) + 10.0**(8.69 - 12) #fraction by number of C, N, and O to H in solar abundances

allowed_mdots = ["bondi", "bhl", "tidal", "gap", "smh", "feedback"]

try:
  from numba import njit
  def cdec(func):
    return njit(func)
except ModuleNotFoundError:
  def cdec(func):
    return func

#### nonlinear structure equations ###

#M is Mstar/Msun
# BAC Equation 8
@cdec
def _solveM(sg, M, Yt):
    M3 = 1.141*sg**2*(1 + 4*Yt/sg)**1.5
    return M - M3

#solve for T given cs, tau, rho, mu
@cdec
def _solveTemp(T, cs0, taufact, rho0, mu, tau):
    Teff4 = T**4/taufact
    prad = 0.5*tau*sbc*Teff4
    pgas = rho0*kboltz*T/mu
    return cs0**2*rho0 - prad - pgas

#BAC Equations A2, A3, and earlier unnumbered equation
@cdec
def _solveT(T, Xn, sg, X, Y):
    Yt = 0.25*(6*X + Y + 2)	# total number of nuclei and electrons per baryon
    Yp = X			# protons per baryon
    Ye = 0.5*(1+X)		# electrons per baryon
    Yn = Xn/14			# number of nitrogen nuclei per baryon?
    sigma = sg*0.25/Yt		# sigma = (1/beta - 1)
    e = (2*sg/Yn)**-1
    nup = 22.42*T**(-1/3) + 7/3
    A = sg*(1 + (nup - 3)*(1+e)/6)**1.5/(Yp*Ye*(1+1/sigma))
    Tc = 2.67*(1 - 0.021*np.log(A) - 0.021/np.log(Xn) + 0.053*np.log(T) )**-3
    return T - Tc

#### Base (linear) accretion rate functions ####
#Assume Bondi accretion
@cdec
def _mdotBondi(M, rho, cs):
    Rb = 2.0*G*M/cs**2
    return np.pi*rho*cs*Rb**2

#Assume Bondi-Hoyle-Lyttleton accretion
@cdec
def _mdotBHL(M, rho, cs, v):
    return 4.0*np.pi*rho*(G*M)**2/(cs**2 + v**2)**1.5

#Assume Stone, Metzger, Haiman 2017 accretion (Eq. 2 and 3)
#N.B. For consistency with the other formulae, I have introduced a factor of 4.
@cdec
def _mdotSMH(M, rho, cs, omega, v, Mbh, h):
    H = h*(omega**2/(G*Mbh))**(-1/3)
    Rh = (G*M/(3*omega**2))**(1/3)
    sig = (v**2 + cs**2 + Rh**2*omega**2)**0.5
    Racc = G*M/sig**2
    return 4.0*np.pi*rho*sig*Racc*np.min(np.array([Racc, H]))

#Bondi or Hill-limited Accretion
@cdec
def _mdotTidal(M, rho, cs, omega):
    Rh = (G*M/(3*omega**2))**(1/3)
    Rb = 2.0*G*M/cs**2
    Racc = np.min(np.array([Rb,Rh]))
    Mdot0 = np.pi*rho*cs*Racc**2
    return Mdot0

#Account for a gap, if one forms. Follows Duffell & Macfadyen 2013, Fung+ 2014, Kanagawa+ 2015, Choski+ 2023.
@cdec
def _mdotGap(M, rho, cs, omega, Mbh, h, alpha):
    Rh = (G*M/(3*omega**2))**(1/3)
    Rb = 2.0*G*M/cs**2
    Racc = np.min(np.array([Rb,Rh]))
    Mdot0 = np.pi*rho*cs*Racc**2

    mu = M/Mbh
    K = mu**2/(h**5*alpha)
    Mdot_out = Mdot0/(1 + 0.04*K)
    return Mdot_out

@cdec
def _LshockV2(dM, v2, Ls, Ledd):
  return 0.5*dM*v2*(1 - Ls/Ledd)/(1 + dM*v2/Ledd)

@cdec
def _Lshock(dM, v2, Ls, Ledd):
  return 0.5*dM*v2

#### nonlinear accretion equations ###

#solve for modified accretion rate as in Cantiello et al. 2021
@cdec
def _solveAcc(M, dM0, Ls, v2, Ledd):
    mmod = dM0*(1 - np.tanh( (Ls + M*v2)/Ledd ) )
    return mmod-M

#solve for modified accretion rate - 2021 form with v_esc reduction
@cdec
def _solveAccV2(M, dM0, Ls, v2, Ledd):
    Lshk = M*v2*(1 - Ls/Ledd)/(1 + M*v2/Ledd)
    mmod = dM0*(1 - np.tanh( (Ls + Lshk)/Ledd ) )
    return mmod-M

#Feedback-limited accretion (Chen, Jiang, Goodman, & Lin 2024)
#Note, it seems like they do not include shock luminosity in the accretion reduction
@cdec
def _solveFeedback(dM, dMb, v2, cr2, Ls, Ledd):
    Lshk = 0.5*dM*v2*(1 - Ls/Ledd)/(1 + dM*v2/Ledd)
    dMr = (Ledd - Ls - Lshk)/cr2
    dMg = (Ledd - Ls - Lshk)/v2
    dMfb = 1.0/(1./dMr + 1./dMg)
    rhs = dM
    lhs = dMb*(1.0 - dM/dMfb)**2.0
    return lhs-rhs

#Feedback-limited accretion - 2024 form with v_esc reduction
@cdec
def _solveFeedbackV2(dM, dMb, v2, cr2, Ls, Ledd):
    Lshk = 0.5*dM*v2*(1 - Ls/Ledd)/(1 + 0.5*dM*v2/Ledd)
    dMr = (Ledd - Ls - Lshk)/cr2
    dMg = Ledd/v2
    dMfb = 1.0/(1./dMr + 1./dMg)
    rhs = dM
    lhs = dMb*(1.0 - dM/dMfb)**2.0
    return lhs-rhs

def _exhaust_event(t, f, rho, cs, X, Y, Z, v, tau, omega, mbh, h, alpha, mdot_method, tkh, fnu, esc_reduce):
    X = f[0]
    return X
_exhaust_event.terminal = True

@cdec
def _timescaleKH(M, R, L):
    tau = 1.5*G*M**2/(R*L*spy)
    return tau

def _runaway_event(t, f, rho, cs, X, Y, Z, v, tau, omega, mbh, h, alpha, mdot_method, tkh, fnu, esc_reduce):
    Ms = np.sum(f)

    if callable(rho):
        rho0 = rho(t)
    else:
        rho0 = rho

    if callable(cs):
        cs0 = cs(t)
    else:
        cs0 = cs

    if callable(v):
        v0 = v(t)
    else:
        v0 = v

    if callable(tau):
        tau0 = tau(t)
    else:
        tau0 = tau

    if callable(h):
        h0 = h(t)
    else:
        h0 = h

    if callable(omega):
        omega0 = omega(t)
    else:
        omega0 = omega

    if callable(X):
        X0 = X(t)
    else:
        X0 = X

    if callable(Y):
        Y0 = Y(t)
    else:
        Y0 = Y

    if callable(Z):
        Z0 = Z(t)
    else:
        Z0 = Z

    MX = f[0]
    MY = f[1]
    MZ = f[2]

    Xs = MX/Ms
    Ys = MY/Ms
    Zs = MZ/Ms

    Yt = 0.25*(6*Xs + Ys + 2)	# nuclei and electrons per baryon, inverse of the total molecular weight
    Ye = (1 + Xs)*0.5		# e per baryon
    sg_g = 7.0			# initial guess for photon entropy
    x, info, err, mesg = fsolve(_solveM, sg_g, args=(Ms, Yt), full_output = 1, xtol = 10**-11)
    sg = x[0]
    sigma = sg*0.25/Yt		# sigma = (1/beta - 1)

    Gamma = 1.0/(4*Yt/sg + 1)	# L/Ledd, BAC equation 9a
    Ledd = 1.2*10**38*Ms/Ye 	# erg /s, BAC equation 9b
    Ls = Gamma*Ledd

    # trying to calculate N mass fraction:
    NH = MX/mh 				# number of H atoms in the star
    NN = cnofrac_solar*NH*(Zs/0.0139) 	#scale by Z/Zsolar
    Xn = NN*14*mh / Ms
    x, info, err, mesg = fsolve(_solveT, 2.0, args=(Xn, sg, Xs, Ys), full_output = 1, xtol = 10**-11)
    Tc = x[0] # central temperature in keV
    Rs = 30.4*Yt*sigma**0.5*(1+sigma)**0.5/Tc #in Rsun
    vesc2 = 2.0*G*Ms*msun/(Rs*rsun)


    if mdot_method == "feedback":
        taufact = 0.5*(0.75*tau0 + 1.0 + 0.5/tau0)
        mu = mh*4.0/(3.0 + 5.0*X0 - Z0)
        guessT = cs0*cs0*mu/(kboltz)    #guess temperature assuming gas

        x, info, err, mesg = fsolve(_solveTemp, guessT, args = (cs0, taufact, rho0, mu, tau0), full_output = 1, xtol = 10**-11)
        T0 = x[0]

        Teff4 = T0**4/taufact
        csrad2 = Teff4*sbc*tau0*0.5/rho0
        dMr = (1-Gamma)*Ledd/csrad2
        dMg = (1-Gamma)*Ledd/vesc2
        dMfb = 1.0/(1.0/dMr + 1.0/dMg)
        #dMb = _mdotBondi(msun*Ms, rho0, cs0)
        dMb = _mdotTidal(msun*Ms, rho0, cs0, omega0)
        dMguess = np.min([dMb, dMg, dMr])
        if esc_reduce:
            x, info, err, mesg = fsolve(_solveFeedbackV2, dMguess, args = (dMb, vesc2, csrad2, Ls, Ledd), full_output = 1, xtol = 10**-11)
        else:
            x, info, err, mesg = fsolve(_solveFeedback, dMguess, args = (dMb, vesc2, csrad2, Ls, Ledd), full_output = 1, xtol = 10**-11)
        Mdot_gain = x[0]

    else:
        if mdot_method == "bondi":
            Mdot_gain = _mdotBondi(msun*Ms, rho0, cs0)
        if mdot_method == "bhl":
            Mdot_gain = _mdotBHL(msun*Ms, rho0, cs0, v0)
        if mdot_method == "smh":
            Mdot_gain = _mdotSMH(msun*Ms, rho0, cs0, omega0, v0, Mbh, h0)
        if mdot_method == "tidal":
            Mdot_gain = _mdotTidal(msun*Ms, rho0, cs0, omega0)
        if mdot_method == "gap":
            Mdot_gain = _mdotGap(msun*Ms, rho0, cs0, omega0, Mbh, h0, alpha)

        if esc_reduce:
            x, info, err, mesg = fsolve(_solveAccV2, Mdot_gain, args=(Mdot_gain, Ls, 0.5*vesc2, Ledd), full_output = 1, xtol = 10**-11)
        else:
            x, info, err, mesg = fsolve(_solveAcc, Mdot_gain, args=(Mdot_gain, Ls, 0.5*vesc2, Ledd), full_output = 1, xtol = 10**-11)
        Mdot_gain = x[0]


    if esc_reduce:
        Lshock = _LshockV2(Mdot_gain, vesc2, Ls, Ledd)
        Ltot = Lshock + Ls
        vesc2 = vesc2*(1 - Ltot/Ledd)
        vesc2 = np.max([vesc2, 10.0**-10]) # prevent sign changes....
        Mdot_loss = (Ltot/vesc2)*(1.0 + np.tanh( 10.0*( Ltot/Ledd - 1) )) #g/s

    else:
        Lshock = _Lshock(Mdot_gain, vesc2, Ls, Ledd)
        Ltot = Lshock + Ls
        Mdot_loss = (Ltot/vesc2)*(1.0 + np.tanh( 10.0*( Ltot/Ledd - 1) )) #g/s

    Mdot_gain*= spy/msun
    tacc = Ms/Mdot_gain

    if callable(tkh):
        tKH = tkh(Ms*msun, Rs*rsun, Ls)
    else:
        tKH = tkh

    return tacc - tKH

_runaway_event.terminal = True

def getExtras(t, f, rho0, cs0, X0, Y0, Z0, v0=None, omega0=None, Mbh=None, h0=None, alpha=None, mdot_method='bondi', tkh=None, fnu=0.1, esc_reduce=False):
    """
    Calculate models of stellar evolution in AGN disks.

    Parameters
    ----------
    t : float
        The time in years.
    f : numpy.ndarray
        A 3-element array holding the mass of hydrogen, helium, and metals consituting the star.
    X0 : float
        Ambient hydrogen mass fraction.
    Y0 : float
        Ambient helium mass fraction.
    Z0 : float
        Ambient metallicity.
    rho0 : float or function, optional.
        Ambient density. Either a constant value (in g/cm^3) or a function of time in years. Defaults to 10^-18 g/cm^3.
    cs0 : float or function, optional.
        Ambient sound speed. Either a constant value (in cm/s) or a function of time in years. Defaults to 10^6 cm/s.
    v0 : float or function. Optional for 'bondi', 'gap,' and 'tidal' accretion, but required for 'bhl' and 'smh' accretion.
        The velocity of the star relative to the ambient medium (in cm/s).
    omega0 : float or function. Optional for 'bondi' and 'bhl' accretion, but required for 'tidal,' 'smh,' or 'gap' accretion.
        Stellar orbital angular velocity. Either a constant value (in 1/s) or a function of time in years.
    h0 : float or function. Optional for 'bondi', 'bhl, or 'tidal' accretion, but required for 'gap' or 'smh' accretion.
        Local disk aspect ratio. Either a constant value (dimensionless) or a function of time in years.
    Mbh : float. Optional for 'bondi', 'bhl, or 'tidal' accretion, but required for 'gap' or 'smh' accretion.
        SMBH mass, a constant value (solar masses).
    alpha : float. Optional for 'bondi', 'bhl, or 'tidal' accretion, but required for 'gap' accretion.
        disk viscosity parameter, a constant value (dimensionless).
    mdot_method : string, optional
        Stellar accretion model. Must be one of ['bondi', 'bhl', 'tidal', 'gap', 'smh']. Defaults to 'bondi'.
    tkh  : float or function, optional
        Not used at present.
    fnu  : float, optional
        The fraction of energy lost via neutrinos during fusion. Defaults to 10%
    esc_reduce  : Boolean, optional
        If True, reduces the escape velocity according to the Eddington ratio. Defaults to False. 

    Returns
    -------
    mdot_gain : float
        The accretion rate onto the star (solar masses / year).
    mdot_loss : float
        The mass loss rate from the star in winds (solar masses / year).
    mdot_burn : float
        The hydrogen burning rate (into helium, in solar masses / year).
    Ls : float
        The intrinsic stellar luminosity from fusion (in solar luminosities).
    Rs : float
        The stellar radius (in solar radii) at each t_eval point.
    Tc : float
        The stellar central temperature (in keV) at each t_eval point.
    """

    if callable(rho0):
        rho0 = rho0(t)

    if callable(h0):
        h0 = h0(t)

    if callable(cs0):
        cs0 = cs0(t)

    if callable(omega0):
        omega0 = omega0(t)

    if callable(v0):
        v0 = v0(t)

    if callable(X0):
        X0 = X0(t)

    if callable(Y0):
        Y0 = Y0(t)

    if callable(Z0):
        Z0 = Z0(t)

    MX = f[0]
    MY = f[1]
    MZ = f[2]

    Ms = MX+MY+MZ

    Xs = MX/Ms
    Ys = MY/Ms
    Zs = MZ/Ms

    Yt = 0.25*(6*Xs + Ys + 2)	# nuclei and electrons per baryon, inverse of the total molecular weight
    Ye = (1 + Xs)*0.5		# e per baryon
    sg_g = 7.0			# initial guess for photon entropy
    x, info, err, mesg = fsolve(_solveM, sg_g, args=(Ms, Yt), full_output = 1, xtol = 10**-11)
    sg = x[0]
    sigma = sg*0.25/Yt		# sigma = (1/beta - 1)

    Gamma = 1.0/(4*Yt/sg + 1)	# L/Ledd, BAC equation 9a
    Ledd = 1.2*10**38*Ms/Ye 	# erg /s, BAC equation 9b
    Ls = Gamma*Ledd

    Mdot_burn = Ls*4*mh/((1.0-fnu)*26.73*mev2erg) 	#g / s, approximation for H burning
    Mdot_burn *= spy/msun 		# msun / yr

    # calculate N mass fraction:
    NH = MX/mh 				# number of H atoms in the star
    NN = cnofrac_solar*NH*(Zs/0.0139) 	#scale by Z/Zsolar
    Xn = NN*14*mh / Ms
    x, info, err, mesg = fsolve(_solveT, 2.0, args=(Xn, sg, Xs, Ys), full_output = 1, xtol = 10**-11)
    Tc = x[0] # central temperature in keV
    Rs = 30.4*Yt*sigma**0.5*(1+sigma)**0.5/Tc #in Rsun
    vesc2 = 2.0*G*Ms*msun/(Rs*rsun)

    if mdot_method == "feedback":
        taufact = 0.5*(0.75*tau0 + 1.0 + 0.5/tau0)
        mu = mh*4.0/(3.0 + 5.0*X0 - Z0)
        guessT = cs0*cs0*mu/(kboltz)    #guess temperature assuming gas

        x, info, err, mesg = fsolve(_solveTemp, guessT, args = (cs0, taufact, rho0, mu, tau0), full_output = 1, xtol = 10**-11)
        T0 = x[0]

        Teff4 = T0**4/taufact
        csrad2 = Teff4*sbc*tau0*0.5/rho0
        dMr = (1-Gamma)*Ledd/csrad2
        dMg = (1-Gamma)*Ledd/vesc2
        dMfb = 1.0/(1.0/dMr + 1.0/dMg)
        #dMb = _mdotBondi(msun*Ms, rho0, cs0)
        dMb = _mdotTidal(msun*Ms, rho0, cs0, omega0)
        dMguess = np.min([dMb, dMg, dMr])
        if esc_reduce:
            x, info, err, mesg = fsolve(_solveFeedbackV2, dMguess, args = (dMb, vesc2, csrad2, Ls, Ledd), full_output = 1, xtol = 10**-11)
        else:
            x, info, err, mesg = fsolve(_solveFeedback, dMguess, args = (dMb, vesc2, csrad2, Ls, Ledd), full_output = 1, xtol = 10**-11)
        Mdot_gain = x[0]

    else:
        if mdot_method == "bondi":
            Mdot_gain = _mdotBondi(msun*Ms, rho0, cs0)
        if mdot_method == "bhl":
            Mdot_gain = _mdotBHL(msun*Ms, rho0, cs0, v0)
        if mdot_method == "smh":
            Mdot_gain = _mdotSMH(msun*Ms, rho0, cs0, omega0, v0, Mbh, h0)
        if mdot_method == "tidal":
            Mdot_gain = _mdotTidal(msun*Ms, rho0, cs0, omega0)
        if mdot_method == "gap":
            Mdot_gain = _mdotGap(msun*Ms, rho0, cs0, omega0, Mbh, h0, alpha)

        if esc_reduce:
            x, info, err, mesg = fsolve(_solveAccV2, Mdot_gain, args=(Mdot_gain, Ls, 0.5*vesc2, Ledd), full_output = 1, xtol = 10**-11)
        else:
            x, info, err, mesg = fsolve(_solveAcc, Mdot_gain, args=(Mdot_gain, Ls, 0.5*vesc2, Ledd), full_output = 1, xtol = 10**-11)
        Mdot_gain = x[0]


    if esc_reduce:
        Lshock = _LshockV2(Mdot_gain, vesc2, Ls, Ledd)
        Ltot = Lshock + Ls
        vesc2 = vesc2*(1 - Ltot/Ledd)
        vesc2 = np.max([vesc2, 10.0**-10]) # prevent sign changes....
        Mdot_loss = (Ltot/vesc2)*(1.0 + np.tanh( 10.0*( Ltot/Ledd - 1) )) #g/s

    else:
        Lshock = _Lshock(Mdot_gain, vesc2, Ls, Ledd)
        Ltot = Lshock + Ls
        Mdot_loss = (Ltot/vesc2)*(1.0 + np.tanh( 10.0*( Ltot/Ledd - 1) )) #g/s

    Mdot_gain*= spy/msun
    Mdot_loss*= spy/msun #msun / yr

    return Mdot_gain, Mdot_loss, Mdot_burn, Ls/lsun, Rs, Tc

def fdot(t, f, rho0, cs0, X0, Y0, Z0, v0=None, tau0=None, omega0=None, Mbh=None, h0=None, alpha=None, mdot_method="bondi", tkh=None, fnu=0.1, esc_reduce=False):
    """
    Calculate models of stellar evolution in AGN disks.

    Parameters
    ----------
    t : float
        The time in years.
    f : numpy.ndarray
        A 3-element array holding the mass of hydrogen, helium, and metals consituting the star.
    X0 : float
        Ambient hydrogen mass fraction.
    Y0 : float
        Ambient helium mass fraction.
    Z0 : float
        Ambient metallicity.
    rho0 : float or function.
        Ambient density. Either a constant value (in g/cm^3) or a function of time in years.
    cs0 : float or function.
        Ambient sound speed. Either a constant value (in cm/s) or a function of time in years.
    tau0 : float or function. Required for 'feedback' accretion, otherwise optional
        Ambient optical depth. Either a constant value or a function of time in years.
    v0 : float or function. Optional for 'bondi', 'gap,' and 'tidal' accretion, but required for 'bhl' and 'smh' accretion.
        The velocity of the star relative to the ambient medium (in cm/s).
    omega0 : float or function. Optional for 'bondi' and 'bhl' accretion, but required for 'tidal,' 'smh,' or 'gap' accretion.
        Stellar orbital angular velocity. Either a constant value (in 1/s) or a function of time in years.
    h0 : float or function. Optional for 'bondi', 'bhl, or 'tidal' accretion, but required for 'gap' and 'smh' accretion.
        Local disk aspect ratio. Either a constant value (dimensionless) or a function of time in years.
    Mbh : float. Optional for 'bondi', 'bhl, or 'tidal' accretion, but required for 'gap' and 'smh' accretion.
        SMBH mass, a constant value (solar masses).
    alpha : float. Optional for 'bondi', 'bhl, or 'tidal' accretion, but required for 'gap' accretion.
        disk viscosity parameter, a constant value (dimensionless).
    mdot_method : string, optional
        Stellar accretion model. Must be one of ['bondi', 'bhl', 'tidal', 'gap', 'smh', 'thermal']. Defaults to 'bondi'.
    tkh  : float or function, optional
        Not used at present, but necessary for consistency with solve_ivp event checking.
    fnu  : float, optional
        The fraction of energy lost via neutrinos during fusion. Defaults to 10%.
    esc_reduce  : Boolean, optional
        If True, reduces the escape velocity according to the Eddington ratio. Defaults to False. 

    Returns
    -------
    df/dt : numpy.ndarray
        An array condating the rate of change of f in years. The first, second, and third elements hold
        the rates of change of the stellar hydrogen, helium, and metal content in solar masses per year.
    """
    if callable(rho0):
        rho0 = rho0(t)

    if callable(h0):
        h0 = h0(t)

    if callable(cs0):
        cs0 = cs0(t)

    if callable(v0):
        v0 = v0(t)

    if callable(tau0):
        tau0 = tau0(t)

    if callable(omega0):
        omega0 = omega0(t)

    if callable(X0):
        X0 = X0(t)

    if callable(Y0):
        Y0 = Y0(t)

    if callable(Z0):
        Z0 = Z0(t)

    #check input validity
    mdot_method = mdot_method.lower()
    if mdot_method not in allowed_mdots:
          print("ERROR: mdot_method must be one of ", allowed_mdots)
          print("Terminating model")
          return -1
    if mdot_method == "bhl":
        if v0 is None:
            print("Error: Bondi-Hoyle-Lyttleton accretion requires setting a velocity ('v0', constant or function, in cgs).")
            print("Terminating model")
            return -1
    if mdot_method == "feedback":
        if tau0 is None:
            print("Error: 'feedback' accretion requires setting an optical depth ('tau0', constant or function).")
            print("Terminating model")
            return -1
    if mdot_method == "tidal":
        if omega0 is None:
            print("Error: tidal limiting requires setting an angular velocity ('omega0', constant or function, in cgs).")
            print("Terminating model")
            return -1
    if mdot_method == "gap":
        if omega0 is None:
            print("Error: gap limiting requires setting an angular velocity ('omega0', constant or function, in cgs).")
            print("Terminating model")
            return -1
        if h0 is None:
            print("Error: gap limiting requires setting an aspect ratio ('h0', constant or function).")
            print("Terminating model")
            return -1
        if alpha is None:
            print("Error: gap limiting requires setting an alpha viscosity ('alpha', constant).")
            print("Terminating model")
            return -1
        if Mbh is None:
            print("Error: gap limiting requires setting an SMBH mass ('Mbh', constant, in solar masses).")
            print("Terminating model")
            return -1
    if mdot_method == "smh":
        if omega0 is None:
            print("Error: Stone-Metzger-Haiman accretion requires setting an angular velocity ('omega0', constant or function, in cgs).")
            print("Terminating model")
            return -1
        if h0 is None:
            print("Error: Stone-Metzger-Haiman accretion requires setting an aspect ratio ('h0', constant or function).")
            print("Terminating model")
            return -1
        if v0 is None:
            print("Error: Stone-Metzger-Haiman accretion requires setting a velocity ('v0', constant or function, in cgs).")
            print("Terminating model")
            return -1
        if Mbh is None:
            print("Error: Stone-Metzger-Haiman requires setting an SMBH mass ('Mbh', constant, in solar masses).")
            print("Terminating model")
            return -1
    if fnu<0.0:
        print("neutrino losses cannot make fusion more efficient, setting to 0")
        fnu = 0.0
    if fnu>1.0:
        print("neutrino losses cannot be more than 1 (100%), setting to 0.1 (10%)")
        fnu = 0.1

    MX = f[0]
    MY = f[1]
    MZ = f[2]

    Ms = MX+MY+MZ

    Xs = MX/Ms
    Ys = MY/Ms
    Zs = MZ/Ms

    Yt = 0.25*(6*Xs + Ys + 2)	# nuclei and electrons per baryon, inverse of the total molecular weight
    Ye = (1 + Xs)*0.5		# e per baryon
    sg_g = 7.0			# initial guess for photon entropy
    x, info, err, mesg = fsolve(_solveM, sg_g, args=(Ms, Yt), full_output = 1, xtol = 10**-11)
    sg = x[0]
    sigma = sg*0.25/Yt		# sigma = (1/beta - 1)

    Gamma = 1.0/(4*Yt/sg + 1)	# L/Ledd, BAC equation 9a
    Ledd = 1.2*10**38*Ms/Ye 	# erg /s, BAC equation 9b
    Ls = Gamma*Ledd

    Mdot_burn = Ls*4*mh/((1.0-fnu)*26.73*mev2erg)	#g / s, approximation for H burning
    Mdot_burn *= spy/msun 		# msun / yr

    # calculate N mass fraction:
    NH = MX/mh 				# number of H atoms in the star
    NN = cnofrac_solar*NH*(Zs/0.0139) 	#scale by Z/Zsolar
    Xn = NN*14*mh / Ms
    x, info, err, mesg = fsolve(_solveT, 2.0, args=(Xn, sg, Xs, Ys), full_output = 1, xtol = 10**-11)
    Tc = x[0] # central temperature in keV
    Rs = 30.4*Yt*sigma**0.5*(1+sigma)**0.5/Tc #in Rsun
    vesc2 = 2.0*G*Ms*msun/(Rs*rsun)

    if mdot_method == "feedback":
        taufact = 0.5*(0.75*tau0 + 1.0 + 0.5/tau0)
        mu = mh*4.0/(3.0 + 5.0*X0 - Z0)
        guessT = cs0*cs0*mu/(kboltz)	#guess temperature assuming gas

        x, info, err, mesg = fsolve(_solveTemp, guessT, args = (cs0, taufact, rho0, mu, tau0), full_output = 1, xtol = 10**-11)
        T0 = x[0]

        Teff4 = T0**4/taufact
        csrad2 = Teff4*sbc*tau0*0.5/rho0
        dMr = (1-Gamma)*Ledd/csrad2
        dMg = (1-Gamma)*Ledd/vesc2
        dMfb = 1.0/(1.0/dMr + 1.0/dMg)
        #dMb = _mdotBondi(msun*Ms, rho0, cs0)
        dMb = _mdotTidal(msun*Ms, rho0, cs0, omega0)
        dMguess = np.min([dMb, dMg, dMr])
        if esc_reduce:
            x, info, err, mesg = fsolve(_solveFeedbackV2, dMguess, args = (dMb, vesc2, csrad2, Ls, Ledd), full_output = 1, xtol = 10**-11)
        else:
            x, info, err, mesg = fsolve(_solveFeedback, dMguess, args = (dMb, vesc2, csrad2, Ls, Ledd), full_output = 1, xtol = 10**-11)
        Mdot_gain = x[0]

    else:
        if mdot_method == "bondi":
            Mdot_gain = _mdotBondi(msun*Ms, rho0, cs0)
        if mdot_method == "bhl":
            Mdot_gain = _mdotBHL(msun*Ms, rho0, cs0, v0)
        if mdot_method == "smh":
            Mdot_gain = _mdotSMH(msun*Ms, rho0, cs0, omega0, v0, Mbh, h0)
        if mdot_method == "tidal":
            Mdot_gain = _mdotTidal(msun*Ms, rho0, cs0, omega0)
        if mdot_method == "gap":
            Mdot_gain = _mdotGap(msun*Ms, rho0, cs0, omega0, Mbh, h0, alpha)

        if esc_reduce:
            x, info, err, mesg = fsolve(_solveAccV2, Mdot_gain, args=(Mdot_gain, Ls, 0.5*vesc2, Ledd), full_output = 1, xtol = 10**-11)
        else:
            x, info, err, mesg = fsolve(_solveAcc, Mdot_gain, args=(Mdot_gain, Ls, 0.5*vesc2, Ledd), full_output = 1, xtol = 10**-11)
        Mdot_gain = x[0]


    if esc_reduce:
        Lshock = _LshockV2(Mdot_gain, vesc2, Ls, Ledd) 
        Ltot = Lshock + Ls
        vesc2 = vesc2*(1 - Ltot/Ledd)
        vesc2 = np.max([vesc2, 10.0**-10]) # prevent sign changes....
        Mdot_loss = (Ltot/vesc2)*(1.0 + np.tanh( 10.0*( Ltot/Ledd - 1) )) #g/s

    else:
        Lshock = _Lshock(Mdot_gain, vesc2, Ls, Ledd) 
        Ltot = Lshock + Ls
        Mdot_loss = (Ltot/vesc2)*(1.0 + np.tanh( 10.0*( Ltot/Ledd - 1) )) #g/s
    

    Mdot_gain*= spy/msun
    Mdot_loss*= spy/msun #msun / yr

    dMx = X0*Mdot_gain - Xs*Mdot_loss - Mdot_burn
    dMy = Y0*Mdot_gain - Ys*Mdot_loss + Mdot_burn
    dMz = Z0*Mdot_gain - Zs*Mdot_loss

    return np.array([dMx, dMy, dMz])

def run(Ms, Xs, Ys, Zs, X0, Y0, Z0, Tend, rho0=10**-18, cs0=10**6, v0=None, tau0=None, omega0=None, h0=None, Mbh=None, alpha=None, mdot_method="bondi", full_output=False, t_eval=None, method='RK54', rtol=1e-6, atol=None, tkh=None, fnu=0.1, check_runaway=False, esc_reduce=False):
    """
    Calculate models of stellar evolution in AGN disks.

    Parameters
    ----------
    Ms : float
        Initial stellar mass (solar masses).
    Xs : float
        Initial stellar hydrogen mass fraction.
    Ys : float
        Initial stellar helium mass fraction.
    Zs : float
        Initial stellar metallicity.
    X0 : float
        Ambient hydrogen mass fraction.
    Y0 : float
        Ambient helium mass fraction.
    Z0 : float
        Ambient metallicity.
    Tend : float
        Target final simulation time.
    rho0 : float or function, optional.
        Ambient density. Either a constant value (in g/cm^3) or a function of time in years. Defaults to 10^-18 g/cm^3.
    cs0 : float or function, optional.
        Ambient sound speed. Either a constant value (in cm/s) or a function of time in years. Defaults to 10^6 cm/s.
    tau0 : float or function. Required for 'feedback' accretion, otherwise optional
        Ambient optical depth. Either a constant value or a function of time in years.
    v0 : float or function. Optional for 'bondi', 'gap,' and 'tidal' accretion, but required for 'bhl' and 'smh' accretion.
        The velocity of the star relative to the ambient medium (in cm/s).
    omega0 : float or function. Optional for 'bondi' and 'bhl' accretion, but required for 'tidal,' 'smh,' or 'gap' accretion.
        Stellar orbital angular velocity. Either a constant value (in 1/s) or a function of time in years.
    h0 : float or function. Optional for 'bondi', 'bhl, or 'tidal' accretion, but required for 'gap' and 'smh' accretion.
        Local disk aspect ratio. Either a constant value (dimensionless) or a function of time in years.
    Mbh : float. Optional for 'bondi', 'bhl, or 'tidal' accretion, but required for 'gap' and 'smh' accretion.
        SMBH mass, a constant value (solar masses).
    alpha : float. Optional for 'bondi', 'bhl, or 'tidal' accretion, but required for 'gap'accretion.
        disk viscosity parameter, a constant value (dimensionless).
    mdot_method : string, optional
        Stellar accretion model. Must be one of ['bondi', 'bhl', 'tidal', 'gap', 'smh']. Defaults to 'bondi'.
    full_output : bool, optional
        If true, outputs additional information (see Returns)
    check_runaway : bool, optional
        If true, terminates the simulation if the accretion timescale becomes shorter than the Kelvin-Helmholtz timescale.
    t_eval : numpy.ndarray, optional
        The times (in years) at which to return outputs.
    method : string, optional.
        The method used by scipy's solve_ivp to solve the ODE system. Defaults to 'RK54'.
    rtol : float, optional.
        The relative error tolerance used by scipy's solve_ivp to solve the ODE system. Defaults to 1e-6.
    atol : float, optional.
        The absolute error tolerance used by scipy's solve_ivp to solve the ODE system. Defaults to rtol/1000.
    tkh  : float or function, optional
        The adopted formula for the Kelvin-Helmholtz timescale in years. As a function, it should take the stellar mass,
        radius, and luminosity (in cgs units) as arguments. Defaults to 1.5*G*M^2/(R*L), the value for an n=3 polutrope.
    fnu  : float, optional
        The fraction of energy lost via neutrinos during fusion. Defaults to 10%

    Returns (default)
    -------
    t : numpy.ndarray
        The times at which solution values are retured.
    m : numpy.ndarray
        The stellar mass (in solar masses) at each t_eval point.
    termination : string
        The reason the evolution concluded.
        'hydrogen exhaution' 	- the star burned through all of its hydrogen fuel.
        'timeout' 		- the simulation reach Tend.
        'runaway' 		- the accretion timescale became shorter than the Kelvin-Helmholtz timescale.

    Returns (full_output=True)
    -------
    Mx : numpy.ndarray
        The stellar mass in hydrogen (in solar masses) at each t_eval point.
    My : numpy.ndarray
        The stellar mass in helium (in solar masses) at each t_eval point.
    Mz : numpy.ndarray
        The stellar mass in metals (in solar masses) at each t_eval point.
    mdot_gain : numpy.ndarray
        The accretion rate onto the star (solar masses / year) at each t_eval point.
    mdot_loss : numpy.ndarray
        The mass loss rate from the star in winds (solar masses / year) at each t_eval point.
    mdot_burn : numpy.ndarray
        The hydrogen burning rate (into helium, in solar masses / year) at each t_eval point.
    Ls : numpy.ndarray
        The intrinsic stellar luminosity from fusion (in solar luminosities ) at each t_eval point.
    Rs : numpy.ndarray
        The stellar radius (in solar radii) at each t_eval point.
    Tc : numpy.ndarray
        The stellar central temperature (in keV) at each t_eval point.

    """

    mdot_method = mdot_method.lower()
    Ms0 = np.array([Xs, Ys, Zs])*Ms

    ## stellar evolution model parameters
    if mdot_method not in allowed_mdots:
          print("ERROR: mdot_method must be one of ", allowed_mdots)
          print("Terminating model")
          return -1
    if mdot_method == "bhl":
        if v0 is None:
            print("Error: Bondi-Hoyle-Lyttleton accretion requires setting a velocity ('v0', constant or function, in cgs).")
            print("Terminating model")
            return -1
    if mdot_method == "feedback":
        if tau0 is None:
            print("Error: 'feedback' accretion requires setting an optical depth ('tau0', constant or function).")
            print("Terminating model")
            return -1
    if mdot_method == "tidal":
        if omega0 is None:
            print("Error: tidal limiting requires setting an angular velocity ('omega0', constant or function, in cgs).")
            print("Terminating model")
            return -1
    if mdot_method == "gap":
        if omega0 is None:
            print("Error: gap limiting requires setting an angular velocity ('omega0', constant or function, in cgs).")
            print("Terminating model")
            return -1
        if h0 is None:
            print("Error: gap limiting requires setting an aspect ratio ('h0', constant or function).")
            print("Terminating model")
            return -1
        if alpha is None:
            print("Error: gap limiting requires setting an alpha viscosity ('alpha', constant).")
            print("Terminating model")
            return -1
        if Mbh is None:
            print("Error: gap limiting requires setting an SMBH mass ('Mbh', constant, in solar masses).")
            print("Terminating model")
            return -1
    if mdot_method == "smh":
        if omega0 is None:
            print("Error: Stone-Metzger-Haiman accretion requires setting an angular velocity ('omega0', constant or function, in cgs).")
            print("Terminating model")
            return -1
        if h0 is None:
            print("Error: Stone-Metzger-Haiman accretion requires setting an aspect ratio ('h0', constant or function).")
            print("Terminating model")
            return -1
        if v0 is None:
            print("Error: Stone-Metzger-Haiman accretion requires setting a velocity ('v0', constant or function, in cgs).")
            print("Terminating model")
            return -1
        if Mbh is None:
            print("Error: Stone-Metzger-Haiman requires setting an SMBH mass ('Mbh', constant, in solar masses).")
            print("Terminating model")
            return -1
    if fnu<0.0:
        print("neutrino losses cannot make fusion more efficient, setting to 0.")
        fnu = 0.0
    if fnu>1.0:
        print("neutrino losses cannot be more than 1 (100%), setting to 0.1 (10%)")
        fnu = 0.1

    if tkh is None:
        tkh = _timescaleKH
    if Mbh is not None:
        Mbh = Mbh*msun

    ## check that ICs are valid
    if Z0 <= 0:
        print("WARNING: This model assumes CNO burning, so running with zero metal accretion may lead to numerical instabilities and unphysical results")
    if Zs <= 0:
        print("Error: This model assumes CNO burning, so the star must have initial Z > 0")
        print("Terminating model")
        return -1

    if check_runaway:
      #check that initial condition does not result in runaway:
      runval = _runaway_event(0.0, Ms0, rho0, cs0, X0, Y0, Z0, v0, tau0, omega0, Mbh, h0, alpha, mdot_method, tkh, fnu)
      if runval < 0:
          print("Initial conditions will lead to runaway accretion")
          print("Terminating model")
          return [-1], [-1], "runaway"

    ## time integration parameters
    if Tend <= 0.0:
        print("Error: requires final simulation time (in years) > 0.")
        print("Terminating model")
        return -1
    else:
        if t_eval is None:
            logtf = np.log10(Tend)
            logti = 4.0
            if logti > logtf - 1: logti = logtf - 1
            t_eval = 10**np.linspace(logti, logtf, 1000)
            t_eval[-1] = Tend

    if atol is None:
        atol = 0.001*rtol

    if check_runaway: termination_events = (_exhaust_event, _runaway_event)
    else: termination_events = (_exhaust_event)

    sol = ivp(fdot, (0, Tend), Ms0, t_eval = t_eval, args = (rho0, cs0, X0, Y0, Z0, v0, tau0, omega0, Mbh, h0, alpha, mdot_method, tkh, fnu, esc_reduce ), events = termination_events, rtol=rtol, atol=atol )
    T = sol.t
    y = sol.y
    m = np.sum(y,axis=0)
    status = sol.status

    if (status == -1):
        termination = "solve_ivp error"
    elif (status == 1):
        tevents = sol.t_events
        #if len(tevents[1]) > 0:  termination = "runaway"
        #if len(tevents[0]) > 0:  termination = "hydrogen exhaustion"
        if len(tevents) > 1: termination = "runaway"
        else: termination = "hydrogen exhaustion"
    else:
        termination = "timeout"
    if full_output:
        Nt = len(m)
        Mx, My, Mz = y[0,:], y[1,:], y[2,:]
        extras = np.empty((Nt, 6))
        for i in range(Nt):
            mdot_gain, mdot_loss, mdot_burn, Ls, Rs, Tc = getExtras(T[i], y[:,i], rho0, cs0, X0, Y0, Z0, v0, omega0, Mbh, h0, alpha, mdot_method, tkh, fnu )
            extras[i,0]=mdot_gain
            extras[i,1]=mdot_loss
            extras[i,2]=mdot_burn
            extras[i,3]=Ls
            extras[i,4]=Rs
            extras[i,5]=Tc
        return T, m, termination, Mx, My, Mz, extras[:,0], extras[:,1], extras[:,2], extras[:,3], extras[:,4], extras[:,5]
    else:
        return T, m, termination
