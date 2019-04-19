# -*- coding: utf-8 -*-
"""
@author: Tong Wu and Jing Guo
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D\

# About Qgate1 class:
#  a class for the 1 qubit gate simulation 

class Qgate1(object):
    def __init__(self):
        #parameters
        self.q = 1.6e-19
        self.hbar  = 1.055e-34
        self.muB = 9.27e-24
        # input
        self.gate=1            # gate type
        self.fz = 200          # zeelman spliting frequency
        self.fb = 2            # decohence frequency
        
        
        self.B1=1.0   # in Tesla, B field in dot1
        self.B2=0.8   # in Telsa, B field in dot2
        self.fz1=2*self.muB*self.B1/(2*np.pi*self.hbar)  # ESR freqency for 1
        self.fz2=2*self.muB*self.B2/(2*np.pi*self.hbar)  # ESR freqency for 1
        
        self.U=6.8   # in meV
        self.tt=10e-3   # in meV
        self.alpha=0.8   # gating efficiency factor
        
        self.flag_master=1  
        self.v=1/(8*1e-9)   # s^-1, spin dephasing rate
        ##
      
        self.gtype = 1
        self.Ns = 1    
        self.Nt = 401
        self.Pr=np.zeros((2*self.Ns,self.Nt))  
        ### depahsing matrix, non-zero terms
        Nh=2*self.Ns    # size of the basis set
        O = [[np.zeros((Nh,Nh)) for i in range(Nh)] for j in range(Nh)] # set up dephasing parameters
        O[1][1][1,1] = 1# diagonal dephasing
        O[0][0][0,0] = 1
        self.O=O 

    def run(self, state, x, y, z, h, s, t):
#        ro0=np.zeros((2*self.Ns, 2*self.Ns))
#        ro0[0,0]=1
        ro0 = np.outer(state, np.conj(state))
        self.ro0 = ro0
        self.ptomography( x, y, z, h, s, t)  #run tomography
        self.Qalg(ro0,x, y, z, h, s, t)

#        self.ftomo  # compute final ro directly from tomography
#        self.visual
    def setH(self, x, y, z, h, s, t):
        # set Hamiltonina matrix for CZ gate

        # Pauli matrices
        S_x = np.array([[0,1], [1,0]])
        S_y = np.array([[0,-1j], [1j,0]])
        S_z = np.array([[1,0], [0,-1]])
        #Hadamard 
        S_h = np.array([[1,1], [1,-1]])/np.sqrt(2)
        #S_gate
        S_s = np.array([[1,0], [0,1j]])
        #T shift
        S_t = np.array([[1,0], [0,np.exp(1j*np.pi/4)]])
        # B.dot(S)
        self.S = x * S_x + y * S_y + z * S_z + h * S_h + s * S_s + t * S_t            
           
        w0=1-self.fb/self.fz#self.fz    # normalized, Zeeman splitting frequency
        self.Heff=w0*1/2*self.S  # unitless, normalized to Zeeman splitting
        tmax = 4*np.pi/w0  # normalized to (1/Zeeman splitting f)
 
        self.tv=np.linspace(0,tmax,self.Nt)  # unitless, normalized to (1/Zeeman splitting f)
        self.w0 = w0
    def Qalg(self,ro0,x, y, z, h, s, t):
        self.setH(x, y, z, h, s, t)   # set Hamiltonian matrix
        self.gam_n=0.1   # unitless, normalized dephasing rate
        
        rop=ro0    # initialization

        self.slvro(self.tv,self.Heff,rop)   # solve Master equation
        
    def slvro(self,tv,Heff,ro0):
       ### output: rop: final density matrix, Pr, probability vs. t,
       ### self.Ut: transform matrix
        rop_ss = []
        rop_ss.append(ro0)
        rop = ro0
        Pr0 = self.Pr
        Pr0[:,0] = np.diag(ro0)
        self.dt = tv[1] - tv[0] 
        self.Heff = Heff
        if self.flag_master==1:  # with decoherence
            for ii_t in range(len(tv)-1):  # time step iteration
                ### Runge-Kutta method for Master Eqn.
                rop = self.rkmethod(rop)
                Pr0[ : , ii_t + 1] = np.real(np.diag(rop)) # Pr is independent of Dirac or Schrodinger when U0 is diagonal
                rop_ss.append(rop)
                if ii_t==101:
                    self.rof=rop
        elif self.flag_master==0:   # ideal without decoherence
            self.Ut=np.expm(-1j*tv.max().dot(Heff))  # the ideal propagator, also used as an output of this function
            rop=self.Ut.dot(ro0).dot(self.Ut.T)  # density matrix in Dirac picture, propagate
            Pr0=[]
            rop_ss.append(rop)
            if ii_t==101:
                self.rof=rop
#        self.rof=rop
        self.rop_ss = rop_ss
        self.Pr = Pr0

    def Integ(self,rop):
        Heff = self.Heff 
        y = -1j * (Heff.dot(rop)-rop.dot(Heff))   # evolution by Heff
        O = self.O  
        N = 2 * self.Ns
        M = N
        for ii_n in range(N):  # iterate over dephasing matrix entries
            for ii_m in range(M):
                L1 = O[ii_n][ii_m]
                y = y + self.gam_n*(L1.dot(rop).dot(L1.T)-1/2*(L1.T.dot(L1).dot(rop)+rop.dot(L1.T).dot(L1))) 
        return y 
    def rkmethod(self,mm):        
        dt = self.dt 
        k1 = self.Integ(mm) 
        
        mm_temp = mm + 1/2 * dt * k1 
        k2 = self.Integ(mm_temp) 
        
        mm_temp = mm + 1/2 * dt * k2 
        k3 = self.Integ(mm_temp) 
        
        mm_temp = mm + dt * k3 
        k4 = self.Integ(mm_temp) 
        
        mm_o=mm + 1/6*(k1 + 2*k2 + 2*k3 + k4) * dt 
        
        return mm_o
    def ptomography(self, x, y, z, h, s, t):  # process tomography

        NF = pow(2,self.Ns)   # size of the Fock space
        self.NF = NF
        Ntomo = pow(NF,2)    # nmber of density matrices        

        temp = []
        for ii in range(Ntomo):
            ro0 = np.zeros((Ntomo,1))
            ro0[ii] = 1 
            ro0 = ro0.reshape(NF,NF).T  
            self.Qalg(ro0, x, y, z, h, s, t)  
            temp.append(self.rof)  
        
        rop = np.vstack([np.hstack([temp[0],temp[2]]),np.hstack([temp[1],temp[3]])])
        
        ### tomography matrices
        I2 = np.eye(2)  
        s = self.S
        K = 1/2 * np.vstack([np.hstack([I2,s]),np.hstack([s,-I2])])
        Tm = K.T.dot(rop).dot(K)    # output tomographe
        self.Tm = Tm  
        self.rotomo = rop   
        
        ### Tomography for the ideal quantum gate for comparison
        U0=self.S    # Ideal gate, quantum tomography for comparison
        temp0 = []
        for ii in range(Ntomo):
            ro0 = np.zeros((Ntomo,1))
            ro0[ii] = 1 
            ro0 = ro0.reshape(NF,NF).T   
            temp0.append(U0.dot(ro0).dot(U0.T)  )
            
        rop0 = np.vstack([np.hstack([temp0[0],temp0[2]]),np.hstack([temp0[1],temp0[3]])])
        Tm0 = K.T.dot(rop0).dot(K)    # output tomographe
        Fidelity=1-0.5*np.sqrt(np.trace((Tm0-Tm).T.dot(Tm0-Tm)))   # Fidelity
        self.Fidelity=Fidelity
        self.Tm0=Tm0     # tomography of the ideal gate
        
    def ftomo(self):  # use tomography results to directly calculation
        E = [] 
        E.append(np.eye(2))
        E.append(self.sx)
        E.append(self.sy)    
        E.append(self.sz)
    
        
        Np = pow(4,self.Ns)    # size of the tomography basis
        Tm = self.Tm   
        NF = self.NF     # size of the Fock space
        ro0 = np.zeros((NF,NF))   
        ro0[0,0] = 1     # initialization
        rof = np.zeros((NF,NF))  
        for ii in range(Np):
            for jj in range(Np):
                rof=rof+Tm[ii,jj]*E[ii]*ro0*(E[jj].T)   
        self.roftomo=rof     # final density matrix from tomography
    def visual(self):
        rof = self.rof # final density matrix
        tv = self.tv
        Pr = self.Pr
        
        Nx=self.NF
        
        #visualize probability density
        self.fig0 = plt.figure()
        plt.plot(tv / 2 /np.pi / self.fz * 1e3, Pr[0,:]*100, linestyle=':',marker='o',label="Up")
        plt.plot(tv / 2 /np.pi / self.fz * 1e3, Pr[1,:]*100, linestyle='-',marker='x',label="Down")
        xx = np.ones(50)*np.pi
        yy = np.linspace(0,100,50)        
        plt.plot(xx / 2 /np.pi / self.fz * 1e3, yy, linestyle=':',label="1 $\\pi$ period")
#        plt.xlim([0, 1])
        plt.ylim([0, 100])
        plt.xlabel('Time (ns)') 
        plt.ylabel('Probability [%]')
        plt.legend()
        
        #tomology of the rael part of density matrix
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        Ntot=Nx*Nx
        xpos, ypos = np.meshgrid(np.linspace(0.75,1.75,Nx), np.linspace(0.75,1.75,Nx))
        xpos=xpos.flatten()
        ypos=ypos.flatten()
        zpos = np.zeros(Ntot)
        dx = 0.5*np.ones(Ntot)
        dy = 0.5*np.ones(Ntot)     
        dz = np.real(rof).flatten()
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]   
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba)
        ax.set_zlabel('Real($\\rho$)',rotation=90)
        ax.set_zlim([0,1])
        ax.set_xticks([1,2])
        ax.set_xticklabels(['0','1'])
        ax.set_yticks([1,2])
        ax.set_yticklabels(['0','1'])
        self.fig1 = fig
        
        #tomology of the img part of density matrix        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        Ntot = Nx*Nx
        xpos, ypos = np.meshgrid(np.linspace(0.75,1.75,Nx), np.linspace(0.75,1.75,Nx))
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(Ntot)
        dx = 0.5*np.ones(Ntot)
        dy = 0.5*np.ones(Ntot)
        dz = np.imag(rof).flatten()
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]   
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba)
        ax.set_zlabel('Imag($\\rho$)',rotation=90)
        ax.set_zlim([0,1])
        ax.set_xticks([1,2])
        ax.set_xticklabels(['0','1'])
        ax.set_yticks([1,2])
        ax.set_yticklabels(['0','1'])
        self.fig2 = fig
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")      
        Nx=4
        Ntot=Nx*Nx
        xpos, ypos = np.meshgrid(np.linspace(0.75,3.75,Nx), np.linspace(0.75,3.75,Nx))
        xpos=xpos.flatten()
        ypos=ypos.flatten()
        zpos = np.zeros(Ntot)        
        dx = 0.5*np.ones(Ntot)
        dy = 0.5*np.ones(Ntot)        
        dz = np.real(self.Tm.flatten())
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]   
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba)
        #ax.set_xlabel("x")
        #ax.set_ylabel("y") 
        ax.set_zlabel('Real($\\rho$)',rotation=90)
        ax.set_zlim([-1,1])
        ax.set_xlim3d(0.5,4.5)
        ax.set_xticks([1,2,3,4])
        ax.set_xticklabels(['I','X','Y','Z'])
        ax.set_ylim3d(0.5,4.5) 
        ax.set_yticks([1,2,3,4])
        ax.set_yticklabels(['I','X','Y','Z'])
        #plt.gca().invert_xaxis()
        self.fig3 = fig

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")      
        Nx=4
        Ntot=Nx*Nx
        xpos, ypos = np.meshgrid(np.linspace(0.75,3.75,Nx), np.linspace(0.75,3.75,Nx))
        xpos=xpos.flatten()
        ypos=ypos.flatten()
        zpos = np.zeros(Ntot)        
        dx = 0.5*np.ones(Ntot)
        dy = 0.5*np.ones(Ntot)        
        dz = np.imag(self.Tm.flatten())
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]   
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba)
        #ax.set_xlabel("x")
        #ax.set_ylabel("y") 
        ax.set_zlabel('Imag($\\rho$)',rotation=90)
        ax.set_zlim([-1,1])
        ax.set_xlim3d(0.5,4.5)
        ax.set_xticks([1,2,3,4])
        ax.set_xticklabels(['I','X','Y','Z'])
        ax.set_ylim3d(0.5,4.5) 
        ax.set_yticks([1,2,3,4])
        ax.set_yticklabels(['I','X','Y','Z'])
        #plt.gca().invert_xaxis()
        self.fig4 = fig