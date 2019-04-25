# -*- coding: utf-8 -*-
"""
@author: Tong Wu and Jing Guo
JG modified on 4/16, correcting several problems
output: tomography Tm, final density matrix, and time-dependent density matrix
Quantum process tomography: Appendix of https://arxiv.org/pdf/1601.07282.pdf
"""
from __future__ import division
#import math
import numpy as np
from scipy import interpolate, linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy as sci
import matplotlib.cm as cm

class Qgate2(object):    
    def __init__(self):
        self.q = 1.6e-19
        self.hbar  = 1.055e-34
        self.muB = 9.27e-24
        
        self.B1=0.05   # in T, B field in dot1
        self.B2=0.04   # in T, B field in dot2

        
        self.U=7.0   # in meV
        self.tt=2e-3   # in meV
        self.udet=6.9   # the detunning energy
        self.flag_master=1  

        self.fd = 1*1e6            # decohence frequency
        ### depahsing matrix, non-zero terms
        self.Ns=4    # size of the basis set
        
        O = [[np.zeros((4,4)) for i in range(4)] for j in range(4)] # set up dephasing parameters
        O[1][1][1,1] = 1# diagonal dephasing
        O[2][2][2,2] = 1
        O[3][3][3,3] = 1
        O[0][0][0,0] = 1
        self.O=O 
        self.data = dict()
        
    def run(self,state):  # the main simulation program
        # input: index of the initial State
        # Output: eigenenergy, time evolution of density matrix, truth table, tomography
        if state == 0: # initial state in [[0,1],[+,-]] basis
            ro0=np.diag([1,0,0,0])
        elif state == 1:
            ro0=np.diag([0,1,0,0])        
        elif state == 2:
            ro0=np.diag([0,0,1,0])        
        elif state == 3:
            ro0=np.diag([0,0,0,1])   
        
        self.slvE() #solve eigen energies  of the detunning gate  
        self.setH()   # set the Hamiltonian matrix
        self.Qgate_operation(ro0) # gate operation time 
        self.Qprocess_operation(ro0) # process of tv time
        self.visual()# python visualization output      
        self.truth_table() # Truth table
        self.tomo() #tomography of a 2qubit gate
        
    
    def slvE(self):  # solve eigen-energy vs. detunning voltage
        # input: B field, tunnel coupling tt, Hubbard U, detunning energy dVg 
        # output: 
        hbar = self.hbar
        q = self.q
        muB = self.muB
        tt = self.tt  # tunnel coupling
        U =self.U  
        Bbar = (self.B1+self.B2)/2
        
        self.fz1=2*self.muB*self.B1/(2*np.pi*self.hbar)  # ESR freqency for QD1
        self.fz2=2*self.muB*self.B2/(2*np.pi*self.hbar)  # ESR freqency for QD2

        dB=abs(self.B1-self.B2)   # B field difference
        dEz=2*muB*dB/q*1e3   # in meV
        
        ### basis: singlet (S11), triplet (T11), singlet (S20)
        H3 = np.array([[0, dEz/2, tt],  [dEz/2, 0, 0],  [tt, 0, U]])  # 3 level Hamiltonian
        
        dVmax=1.2*U   # in mV, maximum detunning voltage
        dVg=dVmax*np.linspace(0.1,1,1000)  # detunning energy, which is dVg if gating efficiency is 1
        
        El = np.zeros((3,len(dVg)))
        wv = np.zeros((len(dVg),3,3))
        for ii_vg in range(len(dVg)):  # loop over detunning energy, 
            dVg_bias = dVg[ii_vg]
            Hb = np.copy(H3)  # without detunning gate voltage
            Hb[2,2] = Hb[2,2]-dVg_bias  # in meV, with gate voltage
            evl,evec = np.linalg.eigh(Hb) 
            El[:,ii_vg] = evl  # energy levels
            wv[ii_vg,:,:] = evec  # wave functions
        
        ### add two triplet levels T(m=1) and T(m=-1)
        Ez = 2*muB*Bbar/q*1e3  # in meV, Zeeman energy 
        El = np.vstack((El,Ez*np.ones((1,len(dVg)))))  # triplet (Tupup)
        El = np.vstack((El,-Ez*np.ones((1,len(dVg)))))  # triplet (Tdowndown)
        ## define energy shift in the reduced range of dVg<U
        indr=np.where(dVg<U)[0] # limit detuning to be <U
        self.dVgr=dVg[indr]
        Eshift= El[0:2,0].reshape((2,-1))-El[0:2,indr]  # energy shift due to applied detunning Vg

        self.dVg=dVg   # gate voltage
        self.El=El    # energy levels
        self.fshift = Eshift.sum(0)*1e-3*q/(2*np.pi*hbar)  # in Hz, frequency shift,, on dVgr

#        wv1 = np.squeeze(wv[:,:,0].T)   # hybridization with S20
#        wv3 = np.squeeze(wv[:,:,2].T)   # hybridization with S20
#        P_s20 = np.zeros((3,len(dVg)))
#        P_s20[0,:] = np.square(abs(wv1[2,:]))   # 1st level S20 component.
#        P_s20[2,:] = np.square(abs(wv3[2,:]))  # probability in S20 State
#        self.wv=wv   # wave function        
#        self.P_s20=P_s20    # weight of mix to S20 state
        
    def setH(self):   # set the effective Hamiltoian and dephasing rate
        # output: self.fCZ (frequency of rotation), self.delay (gate delay)
        # Hg0 and Ug0, the CZ gate Hamiltonian and unitary transform (normalized)
        # Heff and Ueff, the transformed H and U in [[0,1].[+,-]] basis
        self.fCZ= np.interp(self.udet, self.dVgr, self.fshift) # frequency of cz rotation
        self.delay=np.pi/(2*np.pi*self.fCZ)*1e9   # quantum gate delay in ns
        self.gam_n=self.fd/(2*np.pi*self.fCZ) # normalized, dephasing rate to 2*pi*fz
        
        Hg0=np.zeros((4,4))
        Hg0[3,3]=1 # normalized to fshift, effective Hamiltonian of the detunning gate, after combining energy shift
        Had = np.array([[1,1],[1,-1]])/np.sqrt(2)    #definition of 1 bit Hadamard gate
        self.VHad=np.kron(np.eye(2),Had)  # Hadmard on qubit 2
        self.Heff=self.VHad.dot(Hg0.dot(self.VHad))   # transform to [[0,1].[+,-]] basis, normalized effective H
        self.Hg0=Hg0  # The normalizaed Hamiltonian without transform
     
        ## set time series array
        tmax = 4*np.pi  # normalized to (1/2*pi*self.fCZ)
        self.Nt=401
        self.tv=np.linspace(0,tmax,self.Nt)  # unitless, normalized to (1/Zeeman splitting f)
        Ntshort=101  # this short series is for pi rotation gate
        self.tvshort=np.linspace(0,np.pi,Ntshort)  
        
        ### ideal transform matrix
        self.Ueff=linalg.expm(-1j*self.tvshort.max()*(self.Heff))   # with [[0,1].[+,-]] basis    
        self.Ug0=linalg.expm(-1j*self.tvshort.max()*(self.Hg0)) # with [[0,1].[0,1]] basis
    
    def Qgate_operation(self,ro0):   # quantum gate operation, to output sefl.rof
        # output: rof: final density matrix after gate operation
        rof,Pr=self.slvro(self.tvshort,self.Heff,ro0)   # solve Master equation
        self.rof_qg=rof # save to class property
        self.rop_qg=self.rot_ss  # time depedent density matrix  
    
    def Qprocess_operation(self,ro0): # a longer process of tv
        ## input: ro0
        # output: Pr vs tv
        rof,Pr=self.slvro(self.tv,self.Heff,ro0)   # solve Master equation
        self.Pr_tv=Pr  # time depedent density matrix  
        
    def slvro(self,tv,Heff,ro0):
        ### input: ro0: density matrix at t=0, in [[0,1].[+,-]] basis
        ###         tv: time vector
        #           Heff: Hamiltonian matrix to evolve ro in [[0,1].[0,1]] basis
        #   output: rof: density matrix at t=end
        #           Pr(t): diagonal of density matrix vs. t
        ### simulation starts here
        
        
        # initialization
        rop=ro0 
        rot_ss = []  # density matrix vs. time
        rot_ss.append(ro0)
        Pr = np.zeros((len(ro0),len(tv)))
        Pr[:,0] = np.diag(ro0) 
        self.dt = tv[1]-tv[0]           
        for ii_t in range(len(tv)-1):  
        #### Propagate the density operator with Runge-Kutta method for Master Eqn.
            rop = self.rkmethod(rop, Heff)
            Pr[ : , ii_t + 1] = np.real(np.diag(rop)) # Pr is independent of Dirac or Schrodinger when U0 is diagonal
            rot_ss.append(rop)
        self.rot_ss=rot_ss
        rof=rot_ss[-1]
        return rof,Pr
    
    def Integ(self,rop, Heff):
#        Heff = self.Heff 
        y = -1j * (Heff.dot(rop)-rop.dot(Heff))   # evolution by Heff
        O = self.O  
        N = self.Ns
        M = N

        for ii_n in range(N):  # iterate over dephasing matrix entries
            for ii_m in range(M):
                L1 = O[ii_n][ii_m]
                y = y + self.gam_n*(L1.dot(rop).dot(L1.T)-1/2*(L1.T.dot(L1).dot(rop)+rop.dot(L1.T).dot(L1))) 
        return y    

    
    def tomo(self):   # tomography
        Ns=4 
        Nm=pow(Ns,2) 
        roc= []
        for ii in range(Nm):
            ro0=np.zeros((Nm,1))
            ro0[ii]=1  
            ro0=ro0.reshape(Ns,Ns)  
            rof,Pr=self.slvro(self.tvshort,self.Hg0,ro0) # quantum gate calculation
            roc.append(rof)
        
        rop=np.vstack((np.hstack((roc[0], roc[1], roc[2], roc[3])), 
            np.hstack((roc[4], roc[5], roc[6], roc[7])),
            np.hstack((roc[8], roc[9], roc[10], roc[11])), 
            np.hstack((roc[12], roc[13], roc[14], roc[15]))))
        
        ### tomography matrices
        M=np.zeros((Ns,Ns))  
        M[0,0]=1  
        M[3,3]=1 
        M[1,2]=1  
        M[2,1]=1 
        I2=np.eye(2) 
        P=np.kron(I2,np.kron(M,I2)) 
        sx=np.array([[0, 1],[ 1, 0]])  
        sy=np.array([[0, -1j],[ 1j, 0]])
        sz=np.array([[1,0],[0,-1]])
        L=1/4*np.kron(np.kron(sz,I2)+np.kron(sx,sx),np.kron(sz,I2)+np.kron(sx,sx)) 
        K=P.dot(L) 
        Tm=K.T.dot(rop).dot(K)   # output tomographe
        
        ### compute Tm0 for CNOT gate
        U0=self.Ug0   # Ideal CZ gate transform
        roc0 = []
        for ii in range(Nm):
            ro0=np.zeros((Nm,1))  
            ro0[ii]=1  
            ro0=ro0.reshape(Ns,Ns)
            roc0.append(U0.dot(ro0).dot(U0.T) )
        rop0=np.vstack((np.hstack((roc0[0], roc0[1], roc0[2], roc0[3])), 
            np.hstack((roc0[4], roc0[5], roc0[6], roc0[7])),
            np.hstack((roc0[8], roc0[9], roc0[10], roc0[11])), 
            np.hstack((roc0[12], roc0[13], roc0[14], roc0[15]))))# ideal CNOT tomography
        Tm0=K.T.dot(rop0).dot(K)   # ideal CNOT tomography
        Fidelity=1-0.5*np.sqrt(np.trace((Tm0-Tm).T.conj().dot(Tm0-Tm)))  # Fidelity
        self.Tm = Tm
        self.Tm0 = Tm0
        self.Fidelity = Fidelity.real
        self.tomo_plot()
        self.tomo_plot2()
  
    def truth_table(self):  # calculate the truth table of the quantum gate
        # output: truth table in rop_tt, fig in self.fig12
        rop_tt = np.zeros((4,4))
        for ii in range(4):
            ro0=np.zeros((4,4))
            ro0[ii,ii]=1  # initial state
            rof,Pr=self.slvro(self.tvshort,self.Heff,ro0)
            rop_tt[ii,:] = Pr[:,-1]
        self.rop_tt = rop_tt
        Name_state = self.Name_state 
        self.fig12 = plt.figure(9) # initialize the figure
        x=[0,1,2,3]
        y=x
        rop=rop_tt
        xx,yy = np.meshgrid(x,y)
        xx = xx.flatten()+0.25
        yy = yy.flatten()+0.25
        ax = self.fig12.add_subplot(111, projection='3d')
        rop = np.real(rop)
        dx = 0.5 * np.ones_like(xx)
        dy = 0.5 * np.ones_like(yy)
        z = rop.flatten()
        dz = rop.flatten()
        z = np.zeros_like(z)
        
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]          
        ax.bar3d(xx,yy,z,dx,dy,dz, color=rgba, zsort='average')
        
        ax.set_xticks([0.5,1.5,2.5,3.5])
        ax.set_xticklabels([Name_state[0],Name_state[1],Name_state[2],Name_state[3]])
        ax.set_yticks([0.5,1.5,2.5,3.5])
        ax.set_yticklabels([Name_state[0],Name_state[1],Name_state[2],Name_state[3]])
        plt.title('Simulated truth table after gate operation')
        
        
    def rkmethod(self,mm, Heff):         
        dt = self.dt 
        k1 = self.Integ(mm, Heff) 
        
        mm_temp = mm + 1/2 * dt * k1 
        k2 = self.Integ(mm_temp, Heff) 
        
        mm_temp = mm + 1/2 * dt * k2 
        k3 = self.Integ(mm_temp, Heff) 
        
        mm_temp = mm + dt * k3 
        k4 = self.Integ(mm_temp, Heff) 
        
        mm_o=mm + 1/6*(k1 + 2*k2 + 2*k3 + k4) * dt 
        
        return mm_o
    
    def visual(self):   # visualization
        ## outpt self.fig stores the fig object, which can be used in Rappture output
        dVg = self.dVg
        El = self.El*1e3  # in ueV unit
        #energy levels
        self.fig1=plt.figure(1)
        plt.xlim(min(dVg), max(dVg))
        ymax = max(self.tt*5, abs(min(El[:,0]))*2)
        plt.ylim(-ymax,ymax)
        plt.plot(dVg,El[0,:],'-',dVg,El[1,:],dVg,El[2,:],dVg,El[3,:],dVg,El[4,:])
        xx = np.ones(50)*self.udet
        yy = np.linspace(-10,10,50)        
        plt.plot(xx , yy, linestyle=':',label="Operation point")
        plt.legend(loc='upper left')
        plt.xlabel('Detunning [mV]')
        plt.ylabel('E_{mp} [$\\mu$eV]') 
        plt.title('Energy levles')   
        #plt.show()
       
        self.fig9 = plt.figure(7) # time evolution of Probability
        tv=self.tv/(2*np.pi*self.fCZ)*1e9  # in ns, time vector, self.tv normalized to 2*pi*self.fCZ
        tgend=np.pi/(2*np.pi*self.fCZ)*1e9 # in ns, gate time for rotation of pi
        Pr=self.Pr_tv
        plt.plot(tv,Pr[0,:],label=r'$\mathrm{|0}$'+'+'+r'$\rangle$')
        plt.plot(tv,Pr[1,:],label=r'$\mathrm{|0}$'+r'$-\rangle$')
        plt.plot(tv,Pr[2,:],label=r'$\mathrm{|1}$'+'+'+r'$\rangle$')
        plt.plot(tv,Pr[3,:],label=r'$\mathrm{|1}$'+r'$-\rangle$')
        xx = np.ones(50)*tgend
        yy = np.linspace(0,100,50)        
        plt.plot(xx , yy, linestyle=':',label="1 $\\pi$ rotation")
        plt.legend(loc='best')
        plt.xlabel('Time (ns)') 
        plt.ylabel('Probability [%]')
        plt.title('Time evolution')
        plt.ylim([-0.05,1.05])
        #plt.show()

        Name_state = [r'$\mathrm{|0}$'+'+'+r'$\rangle$',r'$\mathrm{|0}$'+r'$-\rangle$',
                    r'$\mathrm{|1}$'+'+'+r'$\rangle$',r'$\mathrm{|1}$'+r'$-\rangle$']
        self.Name_state = Name_state
        
        fig10 = plt.figure(8) # real part of the final density matrix
        x=[0,1,2,3]
        y=x
        rop=self.rof_qg.real
        xx,yy = np.meshgrid(x,y)
        xx = xx.flatten()+0.25
        yy = yy.flatten()+0.25
        ax = fig10.add_subplot(111, projection='3d')

        dx = 0.5 * np.ones_like(xx)
        dy = 0.5 * np.ones_like(yy)
        z = rop.flatten()
        dz = rop.flatten()
        z = np.zeros_like(z)
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]       
        ax.bar3d(xx,yy,z,dx,dy,dz, color=rgba, zsort='average')
        ax.set_zlim([-1,1])      
        ax.set_xticks([0.5,1.5,2.5,3.5])
        ax.set_xticklabels([Name_state[0],Name_state[1],Name_state[2],Name_state[3]])
        ax.set_yticks([0.5,1.5,2.5,3.5])
        ax.set_yticklabels([Name_state[0],Name_state[1],Name_state[2],Name_state[3]])
        ax.set_zlabel('Real($\\rho$)',rotation=90)
        plt.title('Density matrix (Real part) after gate operation')
        self.fig10 = fig10
        
        fig11 = plt.figure(82)  # imaginary part of density matrix
        x=[0,1,2,3]
        y=x
        rop=self.rof_qg.imag
        xx,yy = np.meshgrid(x,y)
        xx = xx.flatten()+0.25
        yy = yy.flatten()+0.25
        ax = fig11.add_subplot(111, projection='3d')

        dx = 0.5 * np.ones_like(xx)
        dy = 0.5 * np.ones_like(yy)
        z = rop.flatten()
        dz = rop.flatten()
        z = np.zeros_like(z)
        
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]   
        
        ax.bar3d(xx,yy,z,dx,dy,dz, color=rgba, zsort='average')
        ax.set_xticks([0.5,1.5,2.5,3.5])
        ax.set_xticklabels([Name_state[0],Name_state[1],Name_state[2],Name_state[3]])
        ax.set_yticks([0.5,1.5,2.5,3.5])
        ax.set_yticklabels([Name_state[0],Name_state[1],Name_state[2],Name_state[3]])
        ax.set_zlim([-1,1])        
        ax.set_zlabel('Imag($\\rho$)',rotation=90)
        plt.title('Density matrix (imag part) after gate operation')
        self.fig11 = fig11
        
    def tomo_plot(self):
        Tm = np.real(self.Tm).flatten()
        Tm0 = np.real(self.Tm0).flatten()
        
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection = "3d")
        ax.view_init(elev=10., azim=-40)
        Nx=16
        Ntot=Nx*Nx
        xpos, ypos = np.meshgrid(np.linspace(0.75,15.75,Nx), np.linspace(0.75,15.75,Nx))
        xpos=xpos.flatten()
        ypos=ypos.flatten()
        zpos = np.zeros(Ntot)
        
        dx = 0.5*np.ones(Ntot)
        dy = 0.5*np.ones(Ntot)
        
        _zpos = Tm0-Tm   # the starting zpos for each bar
        ind=np.where(np.multiply(_zpos,Tm)>0)[0] # index of difference to be shown
        
        dz = Tm
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = np.max(dz)   # get range of colorbars
        min_height = np.min(dz)
        
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]        
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, Tm, color=rgba)
        if len(ind)>0:
            ax.bar3d(xpos[ind], ypos[ind], Tm[ind], dx[ind], dy[ind], _zpos[ind], color='r')        
        ax.set_zlabel('Real($T$)', fontsize=18,rotation=90)
        
        ax.set_xlim3d(0.5,Nx+0.5)
        ax.set_xticks(range(1,Nx+1))
        ax.set_xticklabels(['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ'], fontsize=15)
        ax.set_ylim3d(0.5,Nx+0.5) 
        ax.set_yticks(range(1,Nx+1))
        ax.set_yticklabels(['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ'], fontsize=15)
        ax.set_zlim3d(-.4,.4)
        ax.view_init(20,50)
        plt.tight_layout()
        plt.title('Tomography (Real part) of the CZ Gate after gate operation')
        self.fig20 = fig
        self.data['tomo_real_z'] = dz
        
    def tomo_plot2(self):
        Tm = np.imag(self.Tm).flatten()
        Tm0 = np.imag(self.Tm0).flatten()
        
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection = "3d")
        ax.view_init(elev=10., azim=-40)
        Nx=16
        Ntot=Nx*Nx
        xpos, ypos = np.meshgrid(np.linspace(0.75,15.75,Nx), np.linspace(0.75,15.75,Nx))
        xpos=xpos.flatten()
        ypos=ypos.flatten()
        zpos = np.zeros(Ntot)
        
        dx = 0.5*np.ones(Ntot)
        dy = 0.5*np.ones(Ntot)
        
        _zpos = Tm0-Tm   # the starting zpos for each bar
        ind=np.where(np.multiply(_zpos,Tm)>0)[0] # index of difference to be shown
        
        dz = Tm
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = np.max(dz)   # get range of colorbars
        min_height = np.min(dz)
        
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]        
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, Tm, color=rgba)
        if len(ind)>0:
            ax.bar3d(xpos[ind], ypos[ind], Tm[ind], dx[ind], dy[ind], _zpos[ind], color='r')        
        ax.set_zlabel('Imag($T$)', fontsize=18,rotation=90)
        
        ax.set_xlim3d(0.5,Nx+0.5)
        ax.set_xticks(range(1,Nx+1))
        ax.set_xticklabels(['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ'], fontsize=15)
        ax.set_ylim3d(0.5,Nx+0.5) 
        ax.set_yticks(range(1,Nx+1))
        ax.set_yticklabels(['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ'], fontsize=15)
        ax.set_zlim3d(-.4,.4)
        ax.view_init(20,50)
        plt.tight_layout()
        plt.title('Tomography (Imag part) of the CZ Gate after gate operation')
        self.fig21 = fig        
        
if __name__ == "__main__":  #main function, added by JG for debugging 
    qg=Qgate2()
    State2 = 2
    qg.run(State2)#solve eigen energies
#    qg.Qevolve(State2)# time evolution of the gate
#    qg.visual()# python visualization output     