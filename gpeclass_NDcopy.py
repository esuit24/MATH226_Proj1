'''
This module was adapted from "Bubble.ipynb" from Mateo and "GPE-1D.ipynb" from Chris. It simulates the
Gross-Pitaveskii equation in 1 dimension using dimensionless parameters. It uses the technique of applying
propagators to an imaginary time operator using a second order Suzuki-Trotter expansion to estimate the
chemical potential associated with the given kinetic and potential energy of the system.
'''

# imports
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi
memory_type = np.complex64
import math
from sklearn.preprocessing import normalize
import itertools



class GPEsimulator():
    pi = np.pi
    def __init__(self, dx, dt, time, L = 14, dim = 1, gy = 0, gz = 0, delta = 0, eps = 1, stype = "Default", imp = None):
        # define instances
        self.L = L # Length of the box
        self.dx = dx # grid size
        self.dt = dt # time step size
        self.time = time # total time
        self.dim = dim # dimension: 1, 2, or 3 etc.
        self.gy = gy # gamma y
        self.gz = gz   # gamma z
        self.g_arr = np.array([1, gy, gz]) # array of gammas
        self.delta = delta /np.sqrt(eps) # interaction strength term
        self.eps = eps # epsilon
        self.M = self.time//self.dt # time constant total time / time step
        #self.n = 0
        self.n_arr = None # number of points on the grid
        #self.x =  None # multidimensional array
        self.dk = 0 # momentum step
        #self.kx = None # multidimensional array
        self.gs_pot = None # potential energy grid
        self.gs_ke = None # kinetic energy grid
        self.psi = None # wavefunction of the system
        self.xi = None # x grid
        self.ki = None # momentum grid
        self.stype = stype # type of wavefunction to start with
        self.a0 = 1 # length scale (default = SHO)
        self.imp = imp # import psi
        self.k2 = None

        #self.set_boxwidth()
        self.setxgrid()
        self.setpgrid()
        self.set_energies()


    def setxgrid(self):
        '''
        Sets the x grid based on the member parameters
        '''
        xmax = self.L/2
        self.n = int(2*xmax/self.dx)+1 # number of points on each axis
        n_list = list(itertools.repeat(self.n, self.dim))
        self.n_arr = np.array(n_list)
        axes = []
        for i in range(self.dim):
            axes.append(np.linspace(-xmax + (self.dx/2), xmax - (self.dx/2), self.n))
        axes_arr = np.array(axes)
        self.xi = np.meshgrid(*axes_arr)
        #real_axis = np.linspace(-xmax, xmax, self.n)
        volume_element = self.dx
        #self.x, = np.meshgrid(real_axis)

    def setpgrid2(self):
        '''
        Sets the momentum grid based on the member parameters
        '''
        kmax = pi/self.dx
        self.dk = 2*pi/(self.dx*self.n)
        paxes = []
        for i in range(self.dim):
            paxes.append(np.fft.ifftshift(np.linspace(-kmax, kmax, self.n)))
        paxes_arr = np.array(paxes)
        self.ki = np.meshgrid(*paxes_arr)
        #momentum_axis = np.linspace(-kmax, kmax, self.n)
        #self.kx, = np.meshgrid(momentum_axis
    def setpgrid(self):
        '''
        Sets the momentum grid based on the member parameters
        '''
        kmax = pi/self.dx
        self.dk = 2*pi/(self.dx*self.n)
        paxes = []
        for i in range(self.dim):
            paxes.append(np.fft.fftfreq(self.n,self.dx/(2*np.pi)))
        paxes_arr = np.array(paxes)
        self.ki = np.meshgrid(*paxes_arr)

    def set_energies(self):
        '''
        Sets the potential and kinetic energies of the system based on the parameters
        '''
        self.gs_pot = self.pre_calc_potential()
        self.gs_ke = self.kinetic_energy()

    def set_boxwidth(self):
        case1 = self.a0* 10
        case2 = 0 # replace with TF radius

        if self.dim == 1:
            case2 = np.power(3*self.delta/2, 1/3) *self.a0
        elif self.dim == 2:
            case2 = np.sqrt(2) * np.power(self.delta * self.gy/np.pi, 1/4) * self.a0/min(1, self.gy)
        elif self.dim == 3:
            case2 = np.sqrt(2) * np.power(15 * self.gy * self.gz/(2**(3/2) * 8 * np.pi), 2/5) * self.a0/max(1, self.gy, self.gz)

        self.L = max(case1, 4*case2)



    def kinetic_energy2(self):
        '''
        Defines the kinetic energy for the 1D SHO with m*omega = 1

        Parameters:
        kx - the x wavenumber defining the momentum operator with hbar = 1, m = 1 (p^2/2m => (hk)^2/2m => k^2/2m)

        Returns:
        k2 - the kinetic energy of the SHO system
        '''
        k2 = 0
        for i in range(self.dim):
            k2 += 0.5 * self.eps**2 * (self.ki[i])**2
        #k2 = 0.5*(self.kx**2)
        #k2 = np.fft.fftshift(k2) # fourier shift
        self.k2 = k2
        return k2

    def kinetic_energy(self):
        '''
        Defines the kinetic energy for the 1D SHO with m*omega = 1

        Parameters:
        kx - the x wavenumber defining the momentum operator with hbar = 1, m = 1 (p^2/2m => (hk)^2/2m => k^2/2m)

        Returns:
        k2 - the kinetic energy of the SHO system
        '''
        k2 = 0
        for i in range(self.dim):
            k2 += 0.5 * self.eps**2 * (self.ki[i])**2
        #k2 = 0.5*(self.kx**2)
        #k2 = np.fft.fftshift(k2) # fourier shift
        self.k2 = k2
        return k2

    # External potential energy
    # Start with harmonic oscillator potential with m*omega^2 = 1
    def pre_calc_potential(self):
        '''
        Defines the potential energy of the system using the SHO model

        Returns:
        V - the potential energy of the SHO on the defined x grid
        '''
        V = 0
        for i in range(self.dim):
            V += 0.5 * (self.xi[i]*self.g_arr[i])**2
        #V = (0.5) * self.xi**2
        return V

    ### Define wavefunction helper functions
    def interaction_term(self, dens):
        '''
        Defines the interaction energy of the system

        Parameters:
        dens - the wavefunction density of the system

        Returns:
        Int - the interaction energy
        '''
        Int = self.delta * np.power(self.eps, (self.dim+2)/2) * dens;
        return Int

    # define probability density function
    def density(self):
        '''
        Defines the probability density based on a given wavefunction

        Returns:
        The real probability density (psi* psi)
        '''
        return np.real(self.psi * np.conj(self.psi))

    # normalize the wavefunction psi
    def normalise(self):
        '''
        Normalizes the member wavefunction psi based on a given stepsize for integration

        Updates the member wavefunction based on normalization
        '''
        den = self.density();
        norm = 1/np.sqrt(np.sum(den)*(self.dx**self.dim)) # sum over discrete values * dx = integral over discrete values
        # this is what we want
        self.psi *= norm
        #return psi

    # initialize the wavefunction
    def init_psi(self): # plots constant psi value normalized
        '''
        Initializes a wavefunction at a constant value of psi = 1 and normalizes

        Sets an initial psi as a constant wavefunction
        '''
        #self.psi = np.ones(self.n, dtype=memory_type)
        #n_arr =

        self.psi = np.ones((self.n_arr), dtype = memory_type)
        #print(self.psi)
        self.normalise()
        #print(self.psi)
        #return psi

    def init_psi2(self):
        '''
        Initializes the member wavefunction as
        - "Default" = Constant
        - "TF1" = The Thomas-Fermi wavefunction in 1 dimension
        - "TF2" = The Thomas-Fermi wavefunction in 2 dimensions
        - "NI" = The non-interacting wavefunction (Gaussian)
        - "Import" = a numerical array imported into the class to be used as the wavefunction
        '''
        if self.stype == "Default":
            self.psi = np.ones((self.n_arr), dtype = memory_type)
        elif self.stype == "TF1":
            num = (3/2 * self.delta)**(2/3) - self.xi**2
            den = 2*self.delta*self.a0 * np.ones_like(num)
            for i in range(len(num)):
                if num[i] < 0:
                    num[i] = 0
            self.psi = np.sqrt(num/den)
        elif self.stype == "TF2":
            num = 2*np.sqrt(self.delta * self.gy / np.pi) - (self.xi[0]**2 + self.gy**2 * self.xi[1]**2)
            den = 2*self.delta*self.a0**2 * np.ones_like(num)
            for i in range(len(num)):
                for j in range(len(num[0])):
                    if num[i,j] < 0:
                        num[i,j] = 0

            self.psi = np.sqrt(num/den)
        elif self.stype == "TF3":
            num = 2 * np.power(15 * self.delta * self.gy * self.gz/(8*np.pi*np.power(2,3/2)),2/5) - (self.xi[0]**2 + self.gy**2 * self.xi[1]**2 + self.gz**2 * self.xi[2]**2)
            den = 2 * self.delta * self.a0**3 *np.ones_like(num)
            num[num<0] = 0
            self.psi = np.sqrt(num/den)
        elif self.stype == "NI":
            coef = (pi)**(-0.25*self.dim)
            wf = np.exp( -self.xi[0]**2/2 )
            if self.dim == 1:
                return coef*wf
            else:
                for i in range(self.dim-1):
                    wf *= np.exp( -self.xi[i+1]**2/2 )
            self.psi = coef*wf
        elif self.stype == "Import":
            self.psi = self.imp

        self.psi = self.psi.astype(memory_type)
        self.normalise()


    ### Define Propagators

    # kinetic energy propagator
    def K_propagate(self, dt):
        '''
        Implements the kinetic energy propagator on the given wavefunction (provided in the x basis)

        Parameters:
        dt - time step

        Returns:
        Updates the psi member wavefunction based on the Kinetic Energy propagator
        '''
        # plt.figure()
        # plt.plot(self.psi)
        # plt.title("$\psi$")
        # plt.show()

        fft_psi = np.fft.fftn(self.psi) # fourier transform from spatial space to momentum space
        fft_psi *= np.exp(-1j * dt * self.gs_ke) # apply the kinetic energy operator
        # plt.figure()
        # plt.plot(self.ki[0], np.real(fft_psi))
        # plt.title("Momentum")
        # plt.show()
        self.psi = np.fft.ifftn(fft_psi) # inverse fourier transform from momentum space to spatial space
        #return psi # returns spatial wavefunction
        # plt.figure()
        # plt.plot(self.xi[0], np.real(self.psi) )
        # plt.title("X Space")
        # plt.show()

    # potential energy propagator (incorporating the interaction)
    def VI_propagate(self, dt):
        '''
        Implements the PE propagator on the given wavefunction (all actions performed in the x basis)

        Parameters:
        dt - time step

        Returns:
        Updates the psi member wavefunction based on the potential and interaction propagator
        '''
        self.normalise()
        den = self.density()
        Int = self.interaction_term(den)
        #Int = self.delta
        propagator = np.exp(-1j * (self.gs_pot + Int) * dt) # the potential energy propagator including the interaction prop.
        self.psi *= propagator
        #return psi

    # 2nd order full Suzuki-Trotter propagator
    def step_second_order(self, dt):
        '''
        Applies the second order Suzuki-Trotter expansion to the wavefunction
        Source: https://en.wikipedia.org/wiki/Time-evolving_block_decimation#The_Suzuki%E2%80%93Trotter_expansion

        Parameters:
        dt - time step

        Returns:
        Updates the psi member wavefunction after performing the the Suzuki-Trotter expansion
        '''
        self.VI_propagate(0.5*dt) # half a time step for potential
        self.K_propagate(dt) # full time step for kinetic energy
        self.VI_propagate(0.5*dt)
        #return psi

    # Energy estimator
    def energy(self):
        '''
        Estimates the total energy and each of the constituent energies of the system based on the operators

        Returns:
        E_total - the total energy from kinetic, potential, and interaction energies
        Ek - the magntitude of kinetic energy
        Ev - the magnitude of potential energy
        Ei - the magnitude of interaction energy
        '''
        dF = (self.dx**self.dim)/(np.sqrt(2*np.pi)**self.dim) # for the Fourier transform
        psik = np.fft.fftn(self.psi*dF) # convert to momentum basis
        den = self.density()
        Int = self.interaction_term(den)
        #Int = self.delta
        Ek = np.sum(np.conj(psik)*self.gs_ke*psik)*(self.dk**self.dim)  # kinetic energy
        Ex = np.sum(np.conj(self.psi)*(self.gs_pot + Int)*self.psi)*(self.dx**self.dim)  # interaction and potential energy
        Ev = np.sum(np.conj(self.psi)*self.gs_pot*self.psi)*(self.dx**self.dim)  # potential energy
        Ei = np.sum(np.conj(self.psi)*Int*self.psi)*(self.dx**self.dim)  # interaction energy
        E_total = np.real(Ex + Ek)
        return E_total, np.real(Ek), np.real(Ev), np.real(Ei)

    def simulate(self, num_points):
        '''
        Runs the GPE simulator

        Parameters:
        num_points - the number of points to be used for checking energy convergence

        Returns:
        gs_psi - the resulting wavefunction after simulation
        final_energy - the final energies (total, kinetic, potential, and interaction) after simulation
        ens - the energy distribution as a function of simulation progress
        converge - whether or not the simulation converged and ended early (boolean)
        '''
        ## removed Nsteps as a parameter
        #self.init_psi() # intiialize wavefunction
        self.init_psi2()
        Nsteps = int(self.time/self.dt)
        ens = [] # array of energy values at each time step
        ens_check = self.energy()[0]
        converge = False

        for i in range(Nsteps):
            self.step_second_order(-1.0j*self.dt) # iteratively apply time step to a flat wavefunction psi
            self.normalise() # normalize

            # if i % 10 == 0 and i != 0:
            #     curr = self.energy()[0]
            #     diff = abs(ens_check - curr)
            #     if (diff <= 1e-4):
            #         converge = True
            #         break  # check convergence every time you compute the energy instead
            #         # switch the order of the loops
            #     ens_check = curr

            if i % int(Nsteps/num_points) == 0:
                curr = self.energy()
                #print("Current Energy: ", curr[0])
                #print("Previous Energy: ", ens_check)
                ens.append(curr)
                diff = abs(ens_check - curr[0])
                # make convergence parameter fractional instead of absolute
                if (diff <= 1e-4 and i!= 0):
                    converge = True
                    break  # check convergence every time you compute the energy instead
                    # switch the order of the loops
                ens_check = curr[0]

        # eventually converges
        gs_psi = self.psi
        final_energy = self.energy()
        ens.append(final_energy)

        return gs_psi, final_energy, ens, converge

    # Noninteracting SHO wfn
    def SHO_gswfn(self):
        '''
        The non-interacting simple harmonic oscillator wavefunction (Gaussian) at the ground state

        Returns:
        The result of the Gaussian function applied to each of the points on the discrete x grid
        '''
        ## TODO: incorporate gamma here
        coef = (pi*self.eps)**(-0.25*self.dim)
        wf = np.exp( -self.xi[0]**2/(2*self.eps) )
        if self.dim == 1:
            return coef*wf
        else:
            for i in range(self.dim-1):
                wf *= np.exp( -self.xi[i+1]**2/2 )
        return coef*wf
