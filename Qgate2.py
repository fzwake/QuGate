# -*- coding: utf-8 -*-
"""
@author: Tong Wu and Jing Guo
"""

from __future__ import division
#import math
import numpy as np
from scipy import interpolate 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy as sci
import matplotlib.cm as cm

# About Qgate2 class:
#  a class for the 2 qubit gate simulation 
#  Main methods:
#    solve:        the energy levels simulation
#    Qevolve:      the time evolution simulation
#    DJ1:          DJ algorithm simulation
#    tomo:         tomograph simulation
#    detune:       detuning gate method
#    truth_table:  truth table simulation
#    visual:       visulization 

class Qgate2(object):    
    def __init__(self):
        self.q = 1.6e-19
        self.hbar  = 1.055e-34
        self.muB = 9.27e-24
        
        self.B1=1.0   # in Gauss, B field in dot1
        self.B2=0.8   # in Gauss, B field in dot2

        
        self.U=6.8   # in meV
        self.tt=10e-3   # in meV
        self.alpha=0.8   # gating efficiency factor
        self.flag_master=1  
        self.v=1/(8*1e-9)   # s^-1, spin dephasing rate

        self.fd = 10*1e6            # decohence frequency
        ### depahsing matrix, non-zero terms
        self.Ns=4    # size of the basis set
        O = [[np.zeros((4,4)) for i in range(4)] for j in range(4)] # set up dephasing parameters
        O[1][1][1,1] = 1# diagonal dephasing
        O[2][2][2,2] = 1
        O[3][3][3,3] = 1
        O[0][0][0,0] = 1
        self.O=O 
        self.data = dict()
        
    def solve(self):
        hbar = self.hbar
        q = self.q
        alpha = self.alpha
        muB = self.muB
        tt = self.tt
        U =self.U
        Bbar = (self.B1+self.B2)/2

        self.fz1=2*self.muB*self.B1/(2*np.pi*self.hbar)  # ESR freqency for 1
        self.fz2=2*self.muB*self.B2/(2*np.pi*self.hbar)  # ESR freqency for 1

        dB=abs(self.B1-self.B2)   # B field difference
        dEz=2*muB*dB/q*1e3   # in meV
        
        ### basis: singlet (S11), triplet (T11), singlet (S20)
        H3 = np.array([[0, dEz/2, tt],  [dEz/2, 0, 0],  [tt, 0, U]])  # 3 level Hamiltonian
        
        dVmax=1.2*U/alpha   # in mV, maximum detunning gate voltage
        dVg=dVmax*np.linspace(0.1,1,300)  # voltage difference betwween 2 gates
        
        El = np.zeros((3,len(dVg)))
        wv = np.zeros((len(dVg),3,3))
        for ii_vg in range(len(dVg)):  # loop over gate voltage
            dVg_bias = dVg[ii_vg]
            Hb = np.copy(H3)  # without detunning gate voltage
            Hb[2,2] = Hb[2,2]-alpha*2*dVg_bias/2  # in meV, with gate voltage
            evl,evec = np.linalg.eigh(Hb) 
            El[:,ii_vg] = evl  # energy levels
            wv[ii_vg,:,:] = evec  # wave functions
        
        ### add two triplet levels T(m=1) and T(m=-1)
        Ez = 2*muB*Bbar/q*1e3  # in meV, Zeeman energy 
        El = np.vstack((El,Ez*np.ones((1,len(dVg)))))  # triplet (Tupup)
        El = np.vstack((El,-Ez*np.ones((1,len(dVg)))))  # triplet (Tdowndown)
     
        wv1 = np.squeeze(wv[:,:,0].T)   # hybridization with S20
        wv3 = np.squeeze(wv[:,:,2].T)   # hybridization with S20
        P_s20 = np.zeros((3,len(dVg)))
        P_s20[0,:] = np.square(abs(wv1[2,:]))   # 1st level S20 component.
        P_s20[2,:] = np.square(abs(wv3[2,:]))  # probability in S20 State
        
        pmix = np.linspace(0.02,0.1,9)  # hybridization with S20
        f = interpolate.interp1d(P_s20[0,:],dVg,kind = 'linear',bounds_error =0)
        dVgr = f(pmix)
#        dVgr = interp1(P_s20(1,:),dVg,pmix)  # find gate voltage values
        
        Eshift = np.zeros((2,len(dVg)))
        fgr = np.zeros((2,len(dVgr)))
        
        Eshift[0,:] = El[0,0]-El[0,:]  # energy shift due to applied detunning Vg
        Eshift[1,:] = El[1,0]-El[1,:] 
        fE1 = interpolate.interp1d(dVg,Eshift[0,:],kind = 'slinear')
        fE2 = interpolate.interp1d(dVg,Eshift[1,:],kind = 'slinear')
        fgr[0,:] = fE1(dVgr)*1e-3*q/(2*np.pi*hbar)  # frequency
        fgr[1,:] = fE2(dVgr)*1e-3*q/(2*np.pi*hbar)       
       
        self.dVgr=dVgr  # reduced dVg vector
        self.fgr=fgr   # gate frequency on dVgr
        self.pmix=pmix 
        
        self.dVg=dVg   # gate voltage
        self.El=El    # energy levels
        self.wv=wv   # wave function        
        self.P_s20=P_s20    # wieght of mix to S20 state
        
    def  Qevolve(self, state):  # time evolution
        w02=1-self.fd/self.fz2
        self.w02 = w02
        fz2 = self.fz2   # in Hz, ESR frequency
        fgJ = (self.fgr[:,0]).sum()   #in Hz exchange-determined frequency
        fz2n = fz2/fgJ   # normalized
        sy = (fz2n/2)*np.array([[0 ,-1j],[1j, 0]]) 
        HCZ11 = np.zeros((4,4))  
        HCZ11[3,3] = -1 
        HY2 = np.kron(np.eye(2),sy)    # rotate sita
        HY2b = np.kron(np.eye(2),-sy)   # rotate -sita
        Nt = 401    # number of time steps
        dt = np.pi/Nt   # unitless, normalized time step wt, which is phase
        tv2 = np.linspace(0,np.pi/w02,Nt)  # time, no unit, wJe*t
        tv1 = (1/fz2n)*np.linspace(0,np.pi/2,np.pi/2/dt+1)   #  no unit, wzeeman*t, prefactor for rotational angle         
        tv3 = np.linspace(0,4*np.pi/w02,Nt)  # time, no unit, wJe*t     
        tv32 = np.linspace(0,0.5*np.pi/w02,Nt)  # time, no unit, wJe*t     
        tv = 1/(2*np.pi*fgJ)* np.hstack((tv1, tv1[-1]+tv2[1::], tv1[-1]+tv2[-1]+tv1[1::] ))           
        self.n=self.v/(2*np.pi)/fgJ   # normalized, spin dephasing rate
        
#        rop=np.zeros((4,4))  
#        rop[3,3]=1  # initialization
#        rop[0,0]=1
        if state == 0:
            rop=np.diag([1,0,0,0])
        elif state == 1:
            rop=np.diag([0,1,0,0])        
        elif state == 2:
            rop=np.diag([0,0,1,0])        
        elif state == 3:
            rop=np.diag([0,0,0,1])            
            
        rop1,Pr1=self.slvro(tv1,HY2b,rop) 
        rop2,Pr2=self.slvro(tv2,HCZ11,rop1) 
        rop3,Pr3=self.slvro(tv1,HY2,rop2) 
        
        Pr = np.hstack((Pr1,Pr2[:,1::],Pr3[:,1::]))  # combine results from all stages
        
        # definition of Controlled Z gata and Controlled Not (CX) gate
        HCZ = np.kron(np.array([[1,0],[0,0]]),np.eye(2))
        HCX = np.copy(HCZ)        
        HCZ[2,2] = 1
        HCZ[3,3] = -1        
        HCX[2,3] = 1
        HCX[3,2] = 1
        
        HCZ = HCZ
        HCX = HCX  
        
        H_h1b = np.array([[1,1],[1,-1]])/np.sqrt(2)          #definition of 1 bit Hadamard gate
        H_h2b = np.kron(H_h1b,H_h1b)                 #definition of 2 bit Hadamard gate

        H_h2b3 = np.zeros((4,4))        
        H_h2b3[2:4,2:4] = H_h1b
        H_h2b3[0:2,0:2] = H_h1b
        
#        rop_czh,Pr_czh = self.slvro(tv32,H_h2b,rop) 
        
        self.rop_cz,Pr_cz  = self.slvro_CZ(tv3,HCZ,rop.dot(H_h2b3))
        for i in range(len(Pr_cz.T)):
#            Pr_cz[ : , i] = abs(np.diag(rop).dot(self.rop_cz[i].dot(H_h2b3)))
            Pr_cz[ : , i] = np.real(np.diag(self.rop_cz[i].dot(H_h2b3)))
        
        # the real status, H, CZ, H are all treated as gate 
        rop_czh,Pr_czh1  = self.slvro(tv32,H_h2b3,rop) 
        rop_czh,Pr_czh2  = self.slvro(tv32,HCZ,rop_czh)  
        rop_czh,Pr_czh3  = self.slvro(tv32,H_h2b3,rop_czh)        
        Pr_czh = np.hstack((Pr_czh1,Pr_czh2[:,1::],Pr_czh3[:,1::]))  # combine results from all stages            
        self.tv_czh = np.hstack((tv32, tv32[-1]+tv32[1::], tv32[-1]+tv32[-1]+tv32[1::] ))          
        
        # treat H gate as an ideal one        
        self.rop_cx,Pr_cx=self.slvro(tv3,H_h2b3.dot(HCZ).dot(H_h2b3),rop)               
        ### output results
        self.tv=tv 
        self.tv1=tv1
        self.tv2=tv2
        self.tv3 = tv3 
        self.Pr=Pr
        self.Pr_cz = Pr_cz
        self.Pr_czh = Pr_czh
        self.Pr_cx = Pr_cx    
          
    def slvro_CZ(self,tv,Heff,ro0): # specical sovling function for a CZ gate
        ### input: ro0: density matrix at t=0,
        ###         tv: time vector
        #           H1: Hamiltonian matrix to evolve ro
        #   output: rop: density matrix at t=end
        #           Pr(t): diagonal of density matrix vs. t
        ### simulation starts here
          
        Pr = np.zeros((len(ro0),len(tv)))
        rop = []
        rop.append(ro0)
        Pr[:,0] = np.diag(ro0) 
        self.dt = tv[1]-tv[0]           
        if self.flag_master==1:   # master equation approach
            for ii_t in range(len(tv)-1):  
                # method1
#                drop=-1j*(Heff.dot(rop[ii_t])-rop[ii_t].dot(Heff))*self.dt   # evolution by Heff
#                O = self.O  
#               
#                M=len(O[0]) 
#                n= self.n # dephasing parameters
#                N=len(O)
#                for ii_n in range(N):  # iterate over dephasing matrix entries
#                    for ii_m in range(M):
#                        temp = O[ii_n][ii_m]
#                        drop=drop+self.dt*n*(temp.dot(rop[ii_t]).dot(temp.T)-1/2*(temp.T.dot(temp).dot(rop[ii_t])+rop[ii_t].dot(temp.T).dot(temp)))          
#                rop.append(rop[ii_t]+drop)   # update density matrix
#                Pr[:,ii_t+1] = np.real(np.diag(rop[ii_t+1]))
                # method 2
                rop.append(self.rkmethod(rop[ii_t], Heff))
                Pr[ : , ii_t + 1] = np.real(np.diag(rop[ii_t+1])) # Pr is independent of Dirac
        return rop,Pr             
    
    def slvro(self,tv,Heff,ro0):
        ### input: ro0: density matrix at t=0,
        ###         tv: time vector
        #           H1: Hamiltonian matrix to evolve ro
        #   output: rop: density matrix at t=end
        #           Pr(t): diagonal of density matrix vs. t
        ### simulation starts here
        
        H_h1b = np.array([[1,1],[1,-1]])/np.sqrt(2)          #definition of 1 bit Hadamard gate
        H_h2b = np.kron(H_h1b,H_h1b)                 #definition of 2 bit Hadamard gate

        H_h2b3 = np.zeros((4,4))        
        H_h2b3[2:4,2:4] = H_h1b
        H_h2b3[0:2,0:2] = H_h1b

        rop=ro0  
        Pr = np.zeros((len(ro0),len(tv)))
        
        Pr[:,0] = np.diag(ro0) 
        self.dt = tv[1]-tv[0]           
        if self.flag_master==1:   # master equation approach
            for ii_t in range(len(tv)-1):  
            #### (ii) Propagate the density operator
#                drop=-1j*(Heff.dot(rop)-rop.dot(Heff))*dt   # evolution by Heff
#                O = self.O  
#               
#                M=len(O[0]) 
#                n = self.n # dephasing parameters
#                N=len(O)
#                for ii_n in range(N):  # iterate over dephasing matrix entries
#                    for ii_m in range(M):
#                        temp = O[ii_n][ii_m]
#                        drop=drop+dt*n*(temp.dot(rop).dot(temp.T)-1/2*(temp.T.dot(temp).dot(rop)+rop.dot(temp.T).dot(temp)))          
#                rop=rop+drop   # update density matrix
            ### Runge-Kutta method for Master Eqn.
                rop = self.rkmethod(rop, Heff)
                Pr[ : , ii_t + 1] = np.real(np.diag(rop)) # Pr is independent of Dirac or Schrodinger when U0 is diagonal
                if ii_t == 49: # record the density matrix at half period
                  rop_1 = rop
            ### still habe problem here
            else:   # propagator approach
                pass
        return rop_1,Pr
    
    def Integ(self,rop, Heff):
#        Heff = self.Heff 
        y = -1j * (Heff.dot(rop)-rop.dot(Heff))   # evolution by Heff
        O = self.O  
        N = self.Ns
        M = N

        for ii_n in range(N):  # iterate over dephasing matrix entries
            for ii_m in range(M):
                L1 = O[ii_n][ii_m]
                y = y + self.n*(L1.dot(rop).dot(L1.T)-1/2*(L1.T.dot(L1).dot(rop)+rop.dot(L1.T).dot(L1))) 
        return y     
    
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
    
    def DJ1(self):
        ### D-J algorithm
        # JG, UFL, June 2018.
        # Ref. 1. https://journals.aps.org/prb/pdf/10.1103/PhysRevB.82.184515 
        # f1 (=0): I; f2(=1): Rx^pi; f3 (=x): (I*Ry^(pi/2)Rx^(pi))CZ00(I*Ry^(pi/2))
        # f4 (=1-x): (I*Ry^(-pi/2)Rx^(pi))CZ11(I*Ry^(-pi/2))
        # Ref. 2. : https://www.nature.com/articles/nature25766.pdf
        # f1 (=0): I; f2 (=1): X2^2; f3 (=x): CNOT=Y2*CZ11*Y2bar; f4 (=1-x):Y2bar*CZ00*Y2
        Ui = self.Ugate
        ### Uinp is the input transform matrix being either CZ11 or CZ00
        flag_DJ_Oracle=3  #f1=0  f2=1  f3=x  f4=1-x
        
        sx=np.array([[0,1], [1, 0]])   # unitless, 1 qubit Zeeman matricess
        sy=np.array([[0, -1j], [ 1j, 0]]) 
        sz=np.array([[1, 0], [ 0, -1]]) 
        I2=np.eye(2)   # identity matrix
        
        ts=np.pi/4   # normalized, time, rotation angle
        
        #### Y and Ybar gates
        HY1=np.kron(sy,I2)   # rotate sita 
        HY1b=np.kron(-sy,I2)   # rotate -sita
        HY2=np.kron(I2,sy)    # rotate sita
        HY2b=np.kron(I2,-sy)   # rotate -sita
        HX2=np.kron(I2,sx) 
        
        ##### transform matrices
        UY1=sci.linalg.expm(-1j*HY1*ts)
        UY1b=sci.linalg.expm(-1j*HY1b*ts)
        UY2=sci.linalg.expm(-1j*HY2*ts) 
        UY2b=sci.linalg.expm(-1j*HY2b*ts)
        UX2=sci.linalg.expm(-1j*HX2*ts)
        
        ### two qubit operations
        HCZ00=np.zeros((4,4))  
        HCZ00[0,0]=-1    # unitless, normalized, CZij gates
        HCZ01=np.zeros((4,4))   
        HCZ01[1,1]=-1 
        HCZ10=np.zeros((4,4))    
        HCZ10[2,2]=-1 
        HCZ11=np.zeros((4,4))  
        HCZ11[3,3]=-1 
        
        U1=UY1.dot(UY2b)   # 1st atage
        U3=UY1b.dot(UY2)   # # 3rd stage
        
        td=np.pi  # normalized, time, rotation angle
        if flag_DJ_Oracle==1: # identity, f(0)=f(1)=0
            U2=np.eye(4)    # Identity, f(0)=f(1)=0
        elif flag_DJ_Oracle==2: # X2^2  f2(0)=f2(1)=1
            U2=UX2.dot(UX2) 
        elif flag_DJ_Oracle==3: # CZ11 for CNOT=Y2*CZ11*Y2b, f(0)=0, f(1)=1
            if ~Ui.any():                
                Ui=sci.linalg.expm(-1j*HCZ11*td) # ideal case
            U2=UY2.dot(Ui).dot(UY2b) 
        elif flag_DJ_Oracle==4: # CZ00 for ZCNOT=Y2b*CZ00*Y2, f(0)=1, f(1)=0
            if ~Ui.any():
                Ui=sci.linalg.expm(-1j*HCZ00*td)
            
            U2=UY2b.dot(Ui).dot(UY2) 

        
        Np=4  # number of state
        ro0=np.zeros((Np,Np))  
        ro0[0,0]=1   # initial density matrix
        rop=U3.dot(U2).dot(U1).dot(ro0).dot(U1.conj().T).dot(U2.conj().T).dot(U3.conj().T)   # final density matrix
        self.rop =rop

        
        x=[0,1,2,3]
        y=x
        xx,yy = np.meshgrid(x,y)
        xx = xx.flatten()+0.25
        yy=yy.flatten()+0.25
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        rop = np.real(rop)
        dx = 0.5 * np.ones_like(xx)
        dy = 0.5 * np.ones_like(yy)
        z = rop.flatten()
        dz = rop.flatten()
        z = np.zeros_like(z)
        ax.bar3d(xx,yy,z,dx,dy,dz, color='b', zsort='average')
        
        
#        set(gca,'linewidth',2,'fontsize',20) 
#        set(gca,'xTick', 1:Np)  xlim([0.5 Np+0.5]) 
#        set(gca,'xTickLabel',{'00','01','10','11'}) 
#        set(gca,'yTick', 1:Np)  ylim([0.5 Np+0.5]) 
#        set(gca,'yTickLabel',{'00','01','10','11'}) 
#        zlabel('real(\rho)') 
#        zlim([0 1])
        #plt.show()
        self.fig6 = fig
        self.data = {'x':xx,'y':yy,'z':dz}
        
    def tomo(self):
        U = self.Ugate
        Ns=4 
        Nm=pow(Ns,2) 
        roc= []
        for ii in range(Nm):
            ro0=np.zeros((Nm,1))
            ro0[ii]=1  
            ro0=ro0.reshape(Ns,Ns) 
            roc.append(U.dot(ro0).dot(U.T) )
        
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
        U0=np.array([[1, 0, 0, 0],[ 0, 1, 0, 0],[ 0, 0, 1, 0],[ 0, 0, 0, -1]])   # Ideal CNOT transform
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
        Fidelity=1-0.5*np.sqrt(np.trace((Tm0-Tm).T.dot(Tm0-Tm)))  # Fidelity
        self.Tm = Tm
        self.Tm0 = Tm0
        self.fidelity = Fidelity
        self.tomo_plot()
        self.tomo_plot2()

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
        ind=np.where(np.multiply(_zpos,Tm)>0)
        
        dz = Tm
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = np.max(dz)   # get range of colorbars
        min_height = np.min(dz)
        
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]        
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, Tm, color=rgba)
        ax.bar3d(xpos[ind], ypos[ind], Tm[ind], dx[ind], dy[ind], _zpos[ind], color='r')        
        ax.set_zlabel('Real($\\rho$)', fontsize=18,rotation=90)
        
        ax.set_xlim3d(0.5,Nx+0.5)
        ax.set_xticks(range(1,Nx+1))
        ax.set_xticklabels(['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ'], fontsize=15)
        ax.set_ylim3d(0.5,Nx+0.5) 
        ax.set_yticks(range(1,Nx+1))
        ax.set_yticklabels(['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ'], fontsize=15)
        ax.set_zlim3d(-.4,.4)
        ax.view_init(20,50)
        plt.tight_layout()
        plt.title('Tomography (Real part) of CNOT Gate at $\\pi$ rotation')
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
        ind=np.where(np.multiply(_zpos,Tm)>0)
        
        dz = Tm
        cmap = cm.get_cmap('jet') # Get desired colormap
        max_height = np.max(dz)   # get range of colorbars
        min_height = np.min(dz)
        
        max_height = 1   # get range of colorbars
        min_height = -1        
        # scale each z to [0,1], and get their rgb values
        rgba = [cmap((k-min_height)/(max_height-min_height)) for k in dz]        
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, Tm, color=rgba)
        ax.bar3d(xpos[ind], ypos[ind], Tm[ind], dx[ind], dy[ind], _zpos[ind], color='r')        
        ax.set_zlabel('Real($\\rho$)', fontsize=18,rotation=90)
        
        ax.set_xlim3d(0.5,Nx+0.5)
        ax.set_xticks(range(1,Nx+1))
        ax.set_xticklabels(['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ'], fontsize=15)
        ax.set_ylim3d(0.5,Nx+0.5) 
        ax.set_yticks(range(1,Nx+1))
        ax.set_yticklabels(['II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ'], fontsize=15)
        ax.set_zlim3d(-.4,.4)
        ax.view_init(20,50)
        plt.tight_layout()
        plt.title('Tomography (Imag part) of CNOT Gate at $\\pi$ rotation')
        self.fig21 = fig        
        
    def detune(self):
        Np=5    # size of the basis set
        alpha=0.85  # unitless, gate efficiency factor
        tt=40e-6/np.sqrt(2)   # in eV
        Uhbd=5.94*(2*alpha*1e-3)   # in eV
        BL=1.2  
        BR=0.8   # 1.2 & 0.8T for 125, 12mT and 8mT for 140
        dB=abs(BL-BR)  # in T
        Bbar=0.5*(BL+BR)  # in T
        muB=9.274e-24  
        q=1.6e-19 
        dEz=2*dB*muB/q  # in eV (ud) to (dn) Zeeman energy split
        Ez=4*Bbar*muB/q  # in eV, Zeeman splitting between (uu) and (dd)
        
        ttn=1   # unitless, normalized
        Un=Uhbd/tt  # unitless, normalized
        dEzn=dEz/tt  # unitless, normalized
        Ezn=Ez/tt  # unitless, normalized
        
        #### transient switching calculation
        flag_Ls_vary=1 
        if flag_Ls_vary==0:  # do not treate Ls variation
            Udtn_target=10  # unitless, detunning target
            Udtn_applied=Un-Udtn_target  # unitless
            dVg_applied=tt*Udtn_applied/(2*alpha) # in V, applied detune voltage on 1 gate
        else:  # vix Vg applied
            # another way, fix gate voltage dVg_applied first, derive Udtn_applied
            dVg_applied=5.7736e-03  # for Ls variation, applied Vg is fixed
            Udtn_applied=2*alpha*dVg_applied/tt  #
            
        Ha=np.zeros((Np,Np))  
        Ha[Np-1,Np-1]=-Udtn_applied 
        
        ### Hamiltonian basis (uu,ud,du,dd,S2)
        iuu=1 -1
        iud=2 -1
        idu=3 -1
        idd=4 -1
        is2=Np-1  # indices of ud and du and S2 
        H0=np.diag([Ezn/2, dEzn/2, -dEzn/2, -Ezn/2, Un]) 
        H0[iud,is2]=ttn 
        H0[idu,is2]=ttn  # normalized,due to non-uniform B field
        H0[is2,iud]=ttn  
        H0[is2,idu]=ttn  # due to tunnel coupling
        
        #### compute the eigenstates and enegy without detunning gate voltage.
        [V, Hd]=np.linalg.eig(H0)  # normalized
        
        #### transient simulation
        dt=0.01 

        Nt=1600   # number of time steps, tmax=Nt*dt, tmin=0 
        tv=np.linspace(0,dt*Nt,Nt+1 )
        U1 = np.zeros((Np,Np,Nt+1),'complex')
        U1[:,:,0]=np.eye(Np) 
        phit = np.zeros((Nt+1,1))
        phit[0]=0 
        for ii in range(Nt):
            Htn=H0+Ha  # transient Hamiltonian, in eigenbasis
            U1[:,:,ii+1]=sci.linalg.expm(-1j*Htn*dt).dot(U1[:,:,ii])  # transform. 
            phit[ii+1]=np.arccos(np.cos(-(np.angle(U1[iud,iud,ii+1])+np.angle(U1[idu,idu,ii+1]))))  # angle of rotation
        
        
        ### compute delay
        hbar=1.055e-34  
        q=1.6e-19 
        tunit=hbar/(tt*q)  # in s
        tdelay=Nt*dt*tunit 
        
        ### single gate operations to produce UcZ
        # stage 1: remove B field effect
        Us1=sci.linalg.expm(1j*dt*Nt*np.diag([Ezn/2, dEzn/2, -dEzn/2, -Ezn/2, 0])) # 1st stage single gate operation
        Ueff1=Us1.dot(U1[:,:,-1] )
        # stage 2: combine the angle effect to produce CZ gate
        sud=np.angle(Ueff1[iud,iud])  
        sdu=np.angle(Ueff1[idu,idu]) 
        ### produce CZ11 gate:
        Us2_k=np.kron(np.diag([1, np.exp(-1j*sdu)]), np.diag([1, np.exp(-1j*sud)]))  # for flag_DJ_oracle=3
        ### produce CZ00 gate
        #Us2=kron(diag([exp(-1i*sdu) 1]),diag([exp(-1i*sud) 1]))  # for flag_DJ_oracle=4
        Us2 = np.zeros((5,5),'complex')
        Us2[Np-1,Np-1]=1 
        Us2[0:4,0:4] = Us2_k
        
        Ueff=Us2.dot(Ueff1)  # effective Hamiltonian of transformation)
        self.Ugate=Ueff[0:Np-1,0:Np-1]         
        fig = plt.figure()
        plt.plot(1e9*tv*tunit,phit/np.pi) 
        #plt.show()
        self.fig5= fig
        
    
    def visual(self):
        dVg = self.dVg
        El = self.El
        #energy levels
        self.fig1=plt.figure(1)
        plt.xlim(min(dVg), max(dVg))
        ymax = max(self.tt*5, abs(min(El[:,0]))*2)
        plt.ylim(-ymax,ymax)
        plt.plot(dVg,El[0,:],'-',dVg,El[1,:],dVg,El[2,:],dVg,El[3,:],dVg,El[4,:])
        plt.xlabel('\Delta V_g [mV]')
        plt.ylabel('E_{mp} [meV]') 
        plt.title('Energy levles')   
        #plt.show()
        #visualize probability density
        self.fig2 = plt.figure(2)
        plt.plot(dVg,100*self.P_s20[0,:],label="lowest")
        plt.plot(dVg,100*self.P_s20[2,:],label="S20")
        plt.xlabel('\Delta V_g [mV]') 
        plt.ylabel('P_{De} [%]')
        plt.legend(loc='upper left')
        plt.title('mixing with S20')
        #plt.show()
#        the mixing with the S20 state
        self.fig31 = plt.figure(31)
        dVgr=self.dVgr 
        fgr=self.fgr 
        pmix=self.pmix
        plt.subplot(111)
        plt.plot(dVgr,fgr[0,:])
        plt.plot(dVgr,fgr[1,:])
        plt.xlabel('\DeltaV_g [mV]')
        plt.ylabel('f [Hz]')
        self.fig32 = plt.figure(32)
        plt.subplot(1,1,1)
        plt.plot(dVgr,100*pmix)
        plt.ylim(0,max(100*pmix))
        plt.xlabel('\DeltaV_g [mV]')
        plt.ylabel('P_{de} [%]')
        #plt.show()
        #probability vs. t
        self.fig4 = plt.figure(4) # (DD, UD, DU, UU)
        tv=self.tv
        Pr=self.Pr
        plt.plot(tv,Pr[0,:],label='00')
        plt.plot(tv,Pr[1,:],label='01')
        plt.plot(tv,Pr[2,:],label='10')
        plt.plot(tv,Pr[3,:],label='11')
        plt.legend(loc='best')
        plt.xlabel('Time (ns)') 
        plt.ylabel('Probability [%]')
        plt.title('time evolution')
        #plt.show()
        #probability vs. t of CZ gate
        self.fig7 = plt.figure(5) # (DD, UD, DU, UU)
        tv=self.tv3
        Pr=self.Pr_cz
        plt.plot(tv,Pr[0,:],label=r'$\mathrm{|0}$'+'+'+r'$\rangle$')
        plt.plot(tv,Pr[1,:],label=r'$\mathrm{|0}$'+r'$-\rangle$')
        plt.plot(tv,Pr[2,:],label=r'$\mathrm{|1}$'+'+'+r'$\rangle$')
        plt.plot(tv,Pr[3,:],label=r'$\mathrm{|1}$'+r'$-\rangle$')
        plt.legend(loc='best')
        plt.xlabel('Time (ns)') 
        plt.ylabel('Probability [%]')
        plt.title('time evolution')
        #plt.show()
        #probability vs. t of CNOT gate        
        self.fig8 = plt.figure(6) # (DD, UD, DU, UU)
        tv = self.tv_czh
        Pr=self.Pr_czh
        plt.plot(tv,Pr[0,:],label=r'$\mathrm{|0}$'+'+'+r'$\rangle$')
        plt.plot(tv,Pr[1,:],label=r'$\mathrm{|0}$'+r'$-\rangle$')
        plt.plot(tv,Pr[2,:],label=r'$\mathrm{|1}$'+'+'+r'$\rangle$')
        plt.plot(tv,Pr[3,:],label=r'$\mathrm{|1}$'+r'$-\rangle$')
        plt.legend(loc='center left')
        plt.xlabel('Time (ns)') 
        plt.ylabel('Probability [%]')
        plt.title('time evolution')
        plt.ylim([-0.05,1.05])
        #probability vs. t of CNOT gate        
        self.fig9 = plt.figure(7) # (DD, UD, DU, UU)
        tv=self.tv3
        Pr=self.Pr_cx
        plt.plot(tv,Pr[0,:],label=r'$\mathrm{|0}$'+'+'+r'$\rangle$')
        plt.plot(tv,Pr[1,:],label=r'$\mathrm{|0}$'+r'$-\rangle$')
        plt.plot(tv,Pr[2,:],label=r'$\mathrm{|1}$'+'+'+r'$\rangle$')
        plt.plot(tv,Pr[3,:],label=r'$\mathrm{|1}$'+r'$-\rangle$')
        xx = np.ones(50)*tv[50]
        yy = np.linspace(0,100,50)        
        plt.plot(xx , yy, linestyle=':',label="1 $\\pi$ period")
        plt.legend(loc='best')
        plt.xlabel('Time (ns)') 
        plt.ylabel('Probability [%]')
        plt.title('time evolution')
        plt.ylim([-0.05,1.05])
        #plt.show()

        Name_state = [r'$\mathrm{|0}$'+'+'+r'$\rangle$',r'$\mathrm{|0}$'+r'$-\rangle$',
                    r'$\mathrm{|1}$'+'+'+r'$\rangle$',r'$\mathrm{|1}$'+r'$-\rangle$']
        self.Name_state = Name_state
        
        fig10 = plt.figure(8)
        x=[0,1,2,3]
        y=x
        rop=self.rop_cx
        xx,yy = np.meshgrid(x,y)
        xx = xx.flatten()+0.25
        yy = yy.flatten()+0.25
        ax = fig10.add_subplot(111, projection='3d')
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
        ax.set_zlim([-1,1])      
        ax.set_xticks([0.5,1.5,2.5,3.5])
        ax.set_xticklabels([Name_state[0],Name_state[1],Name_state[2],Name_state[3]])
        ax.set_yticks([0.5,1.5,2.5,3.5])
        ax.set_yticklabels([Name_state[0],Name_state[1],Name_state[2],Name_state[3]])
        ax.set_zlabel('Real($\\rho$)',rotation=90)
        plt.title('Density matrix (Real part) of CNOT Gate at $\\pi$ rotation')
        self.fig10 = fig10
        
        fig11 = plt.figure(82)
        x=[0,1,2,3]
        y=x
        rop=self.rop_cx
        xx,yy = np.meshgrid(x,y)
        xx = xx.flatten()+0.25
        yy = yy.flatten()+0.25
        ax = fig11.add_subplot(111, projection='3d')
        rop = np.imag(rop)
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
        plt.title('Density matrix (imag part) of CNOT Gate at $\\pi$ rotation')
        self.fig11 = fig11
        
    def truth_table(self):
        state = [0,1,2,3]
        rop_tt = np.zeros((4,4))
        for i in range(4):
            self.Qevolve(state[i])#
#            rop_tt[i,:] = np.diag(self.rop_cx)
            rop_tt[i,:] = self.Pr_cx[:,50]
        self.rop_tt = rop_tt
        Name_state = self.Name_state 
        fig = plt.figure(9)
        x=[0,1,2,3]
        y=x
        rop=rop_tt
        xx,yy = np.meshgrid(x,y)
        xx = xx.flatten()+0.25
        yy = yy.flatten()+0.25
        ax = fig.add_subplot(111, projection='3d')
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
        plt.title('Truth table of CNOT Gate at $\\pi$ rotation')
        self.fig12 = fig