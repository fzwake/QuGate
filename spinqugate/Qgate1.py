# -*- coding: utf-8 -*-
"""
@author: Tong Wu and Jing Guo
JG modified on 4/16, correcting several problems
output: tomography Tm, final density matrix, and time-dependent density matrix
Quantum process tomography: Appendix of https://arxiv.org/pdf/1601.07282.pdf
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
        #self.q = 1.6e-19
        #self.hbar  = 1.055e-34
        #self.muB = 9.27e-24
        # input
        self.gate=1            # gate type
        self.fz = 28          # zeelman spliting frequency, JG removed, from input
        self.gam = 10            # decohence frequency, JG removed, from input
        
        self.flag_master=1  

        self.gtype = 1
        self.Ns = 1    
        self.Nt = 401
        self.Pr=np.zeros((2*self.Ns,self.Nt)) # output probabiity vs. time 
        ### depahsing matrix, non-zero terms
        Nh=2*self.Ns    # size of the basis set
        L = [[np.zeros((Nh,Nh)) for i in range(Nh)] for j in range(Nh)] # set up dephasing parameters
        L[1][1][1,1] = 1# diagonal dephasing
        L[0][0][0,0] = 1
        self.L=L 

    def run(self, state, x, y, z, h, s, t):  # main simulation program
#        ro0=np.zeros((2*self.Ns, 2*self.Ns))
#        ro0[0,0]=1
        self.setH(x, y, z, h, s, t)   # set Hamiltonian matrix
   
        self.gam_n=self.gam/(2*np.pi*self.fz)   # unitless, normalized dephasing rate to t*pi*fz
        
        ro0 = np.outer(state, np.conj(state))
        self.ro0 = ro0
        self.ptomography( x, y, z, h, s, t)  #run tomography
        self.Qgate_operation(ro0) # gate operation
        self.slvro(self.tv,self.Heff,ro0) # obtain time-dependent evolution

#        self.visual
    def setH(self, x, y, z, h, s, t):

        #  B^ dot Pauli matrix vector, for computing effective Hamiltonian
        S_x = np.array([[0,1], [1,0]]) # B along x
        S_y = np.array([[0,-1j], [1j,0]]) # B along y
        S_z = np.array([[1,0], [0,-1]]) # B along z
        #Hadamard, B field along XZ, pi rotation
        S_h = (S_x+S_z)/np.sqrt(2)
        #S_gate, B field along Z, pi/2 rotation
        S_s = S_z
        #T shift, B field along Z, pi/4 rotation
        S_t = S_z
        # B.dot(S)
        BdS = x * S_x + y * S_y + z * S_z + h * S_h + s * S_s + t * S_t            
           
        w0=1    #self.fz    # normalized, to 2*pi*fz
        self.w0 = w0
        ### effective Hamiltonian due to magnetic field
        self.Heff=w0*1/2*BdS
     
        ## set time series array
        
        tmax = 4*np.pi  # normalized to (1/2*pi*Zeeman splitting f)
        self.tv=np.linspace(0,tmax,self.Nt)  # unitless, normalized to (1/Zeeman splitting f)
        Ntshort=101  # this short series is for pi rotation gate
        
        if x==1 or y==1 or z==1: # X, Y, Z gates are pi rotation
            Rphase=1 # rotation phase in pi
        elif h==1: # Hardmard is pi rotation
            Rphase=1
        elif s==1: # S gate is pi/2 rotation
            Rphase=0.5
        elif t==1: # T gate is pi/4 rotation
            Rphase=0.25  
        self.tvshort=np.linspace(0,np.pi*Rphase,Ntshort)  
        
        ### ideal transform matrix
        sita=np.pi*Rphase  # rotation angle
        self.U0=np.cos(sita/2)*np.eye(2)-1j*np.sin(sita/2)*BdS
        self.S = BdS
    def Qgate_operation(self,ro0):   # quantum gate operation, to output sefl.rof
        rof=self.slvro(self.tvshort,self.Heff,ro0)   # solve Master equation
        self.rof=rof # save to class property
        self.rop_qg=self.rop_ss
        
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
                
        elif self.flag_master==0:   # ideal without decoherence
            self.Ut=np.expm(-1j*tv.max().dot(Heff))  # the ideal propagator, also used as an output of this function
            rop=self.Ut.dot(ro0).dot(self.Ut.T)  # density matrix in Dirac picture, propagate
            Pr0=[]
            rop_ss.append(rop)
        rof=rop   # final density matrix   
        
        self.rop_ss = rop_ss  # evolution of the density matrix
        self.Pr = Pr0
        return rof  # return
    
    def Integ(self,rop):
        Heff = self.Heff 
        y = -1j * (Heff.dot(rop)-rop.dot(Heff))   # evolution by Heff
        L = self.L  
        N = 2 * self.Ns
        M = N
        for ii_n in range(N):  # iterate over dephasing matrix entries
            for ii_m in range(M):
                L1 = L[ii_n][ii_m] # dephasing operator
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
        # corrected by JG, tomography needs to use a pi angle rotation

        Mat = []   # individual output density matrix list for tomography
        for ii in range(Ntomo):
            ro0 = np.zeros((Ntomo,1))
            ro0[ii] = 1 
            ro0 = ro0.reshape(NF,NF) 
            rof=self.slvro(self.tvshort,self.Heff,ro0)
            Mat.append(rof)  
        
        rotomo = np.vstack([np.hstack([Mat[0],Mat[1]]),np.hstack([Mat[2],Mat[3]])])
        
        ### tomography matrices
        I2 = np.eye(2)  
        sx = np.array([[0,1], [1,0]])
        K = 1/2 * np.vstack([np.hstack([I2,sx]),np.hstack([sx,-I2])])
        Tm = K.dot(rotomo).dot(K)    # output tomography
        self.Tm = Tm        # tomography of the physical gate 
        self.rotomo = rotomo   
        
        ### Tomography for the ideal quantum gate for comparison
        U0=self.U0    # Ideal gate, quantum tomography for comparison
        Mat0 = []
        for ii in range(Ntomo):
            ro0 = np.zeros((Ntomo,1))
            ro0[ii] = 1 
            ro0 = ro0.reshape(NF,NF)   
            Mat0.append(U0.dot(ro0).dot(U0.T.conj()))
            
        rotomo0 = np.vstack([np.hstack([Mat0[0],Mat0[1]]),np.hstack([Mat0[2],Mat0[3]])])
        Tm0 = K.dot(rotomo0).dot(K)    # output tomographe
        Fidelity=1-0.5*np.sqrt(np.trace((Tm0-Tm).T.conj().dot(Tm0-Tm))).real   # Fidelity
        self.Fidelity=Fidelity
        self.Tm0=Tm0     # tomography of the ideal gate

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
        ax.set_zlim([-1,1])
        ax.set_xticks([1,2])
        ax.set_xticklabels(['0','1'])
        ax.set_yticks([1,2])
        ax.set_yticklabels(['0','1'])
        plt.title('Density matrix (Real part) after gate operation')
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
        ax.set_zlim([-1,1])
        ax.set_xticks([1,2])
        ax.set_xticklabels(['0','1'])
        ax.set_yticks([1,2])
        ax.set_yticklabels(['0','1'])
        plt.title('Density matrix (Imag part) after gate operation')
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
        plt.title('Tomography (Real part) of the CZ Gate after gate operation')
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
        plt.title('Tomography (Imag part) of the CZ Gate after gate operation')
        self.fig4 = fig
        
if __name__ == "__main__":  #main function, added by JG for debugging 
    qg=Qgate1()
    ro0=np.asarray([[1,0],[0,0]])
    psi0=1./np.sqrt(2)*np.asarray([1.,1.])
    qg.run(psi0,0,0,1,0,0,0)
