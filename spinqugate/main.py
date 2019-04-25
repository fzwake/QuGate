# -*- coding: utf-8 -*-
"""
@author: Tong Wu and Jing Guo
# problem corrected by JG 4/16, line 101 self.fd
"""
# The code generates a simulation tool based on Rappture for Quantum Gate.
# 
# Tong Wu and Jing Guo in Aug 2018
#
from Qgate2 import Qgate2
from Qgate1 import Qgate1
import Rappture
import sys
import numpy as np

from figure import *

#method to get the parameters from rappture 
def get_label(driver, where, name, units="T"):
    v = driver.get("{}.({}).current".format(where, name))
    if (v != None):
        print "{} = {}\n".format(name, v)
    else:
        print "Failed to retrieve {}{} from driver.xml\n".format(where,name)

    v2 = Rappture.Units.convert(v, to=units, units="off")
    return (v, v2)

# main code
if __name__ == "__main__":
    # define the parameters
    h=6.626e-34
    mu = 9.27e-24
    q = 1.6e-19
    
    #define a rappture library
    driver1 = Rappture.library(sys.argv[1])
    
    # get the model selection flag
    flag_model=driver1.get("input.phase(two).(main).loader.current")
    print('model: '+flag_model)
    
    if flag_model=='1. Single qubit rotation gate':
      driver1 = Rappture.library(sys.argv[1])
      #get the input parameters from the input interface in the library  
#      _,rotate_theta = get_label(driver1, "input.(two).(main).(000).(s2).current.parameters", "theta","deg")
#      _,rotate_phi = get_label(driver1, "input.(two).(main).(000).(s2).current.parameters", "phi","deg")
#      _,rotate_alpha = get_label(driver1, "input.(two).(main).(000).(s2).current.parameters", "alpha","deg")

#      rotate_theta,_ = get_label(driver1, "input.(two).(main).(000).(s2).current.parameters", "theta")
#      rotate_phi,_ = get_label(driver1, "input.(two).(main).(000).(s2).current.parameters", "phi")
#      rotate_alpha,_ = get_label(driver1, "input.(two).(main).(000).(s2).current.parameters", "alpha")
   
      gate,_ = get_label(driver1, "input.(two).(main).(tt).(s1).current.parameters", "gate") 
      S_ini,_ = get_label(driver1, "input.(two).(main).(tt).(s1).current.parameters", "S_ini") 
      _,B = get_label(driver1, "input.(two).(main).(tt).(s1).current.parameters", "B", "G") 
      _,T = get_label(driver1, "input.(two).(main).(tt).(s1).current.parameters", "T2", "us") 

      fz = 2*mu*B/h*1e-4*1e-6  # in MHz
      gam = 1/T
      
      #convert the parameter to input of single qubit
      magnetic_x = 0
      magnetic_y = 0
      magnetic_z = 0
      magnetic_s = 0  
      magnetic_h = 0
      magnetic_t = 0 
      magnetic_phi = 0   
      if gate == 'X direction (for X gate)':
        magnetic_x = 1
      elif gate == 'Y direction (for Y gate)':
        magnetic_y = 1       
      elif gate == 'Z direction (for Z gate)':
        magnetic_z = 1 
      elif gate == 'Z direction (for S gate)':
        magnetic_s = 1 
      elif gate == 'for Hadamard gate':
        magnetic_h = 1   
      elif gate == 'Z direction (for T gate)':
        magnetic_t = 1 
  
      if S_ini == 'Z direction':
        state = np.array([1,0])
      elif S_ini == 'X direction':
        state = np.array([1,1])*np.sqrt(2)/2   
      elif S_ini == 'Y direction':
        state = np.array([1,1j])*np.sqrt(2)/2   
      elif S_ini == '-Z direction':
        state = np.array([0,1])
      elif S_ini == '-X direction':
        state = np.array([1,-1])*np.sqrt(2)/2   
      elif S_ini == '-Y direction':
        state = np.array([1,-1j])*np.sqrt(2)/2    
        
      # indicate the processing status
      Rappture.Utils.progress(25,"Solving single qubit...")      

      #define the single qubit class and visulize the result
      qgate = Qgate1() # initialize
      qgate.fz = fz
      qgate.gam = gam
      
      qgate.run(state, magnetic_x, magnetic_y, magnetic_z, magnetic_h, magnetic_s, magnetic_t)     # run the main code
      qgate.visual()  #visulize

      #output the figures to the result of rappture 
      fig_TE1(qgate.fig0,11,driver1)
      fig = [qgate.fig1,qgate.fig2,qgate.fig3,qgate.fig4]
      name = ['Density matrix (real part)','Density matrix (imag part)','Tomography (real)','Tomography (imag)']
      for i in range(4):
        fig_put(fig[i],name[i],i,driver1)

      # main simulation code for single spin
      
      #bloch_arbitrary_rotate(state, rotate_theta/np.pi, rotate_phi/np.pi, rotate_alpha/np.pi, driver1)
      #bloch_magnetic_field(magnetic_x, magnetic_y, magnetic_z, magnetic_s, magnetic_h, magnetic_t, magnetic_phi, driver1)
      #bloch_time_evo(magnetic_x, magnetic_y, magnetic_z, magnetic_h, magnetic_s, magnetic_t, magnetic_phi, driver1)
      bloch_time_evo2(qgate.rop_qg, magnetic_x, magnetic_y, magnetic_z, magnetic_h, magnetic_s, magnetic_t, qgate.S, driver1)
                   
      Rappture.Utils.progress(99,"Logging...")

      separator = '\n'+'='*60    #separator used in rappture output
      separator2 = '\n'+'-'*60   #separator2 used in rappture output
      
      #output contetnts: the input parameters and some of the result to a log file      
      output=separator+'\nZeeman spliting frequency: '+str(fz)+'MHz'+separator2+'\nGate type:\n  '+gate+separator2+'\nFidelity='+str(np.real(qgate.Fidelity))+separator2+'\nDensity matrix:\n\n'+str(qgate.rof)+separator2+'\nTomography data:\n\n'+str(qgate.Tm)+separator
      
      # push the output file to the difined library
      driver1.put('output.log()',output,append=0)      
      driver1.put('output.log().about.label','Quantum gate output data',append=0)
            
      Rappture.result(driver1) # output the results 
      sys.exit()      #exit
    
    elif flag_model=='2. Two qubit gate': 
      driver = Rappture.library(sys.argv[1])
      #get the input parameters from the input interface in the library  
      B1,B1_2 = get_label(driver, "input.(two).(main).(tt).structure(s1).current.parameters","B1","G")
      B2,B2_2 = get_label(driver, "input.(two).(main).(tt).structure(s1).current.parameters","B2","G")
      U,U_2 = get_label(driver, "input.(two).(main).(tt).structure(s1).current.parameters", "U", "meV")
      t,t_2 = get_label(driver, "input.(two).(main).(tt).structure(s1).current.parameters", "t", "meV")
      _,T2 = get_label(driver1, "input.(two).(main).(tt).(s1).current.parameters", "T2", "us") 
      alpha,alpha_2 = get_label(driver, "input.(two).(main).(tt).structure(s1).current.parameters", "detuning energy","meV")
      S_ini,_ = get_label(driver, "input.(two).(main).(tt).(s1).current.parameters", "S_ini") 

      #initalize the QuGate object
      dev = Qgate2()
      
      # indicate the processing status
      Rappture.Utils.progress(0,"Starting...")
      
      #pass the input parameters to the class object

      dev.U  = U_2
      dev.fd = 1/T2*1e6
      dev.B1 = B1_2*1e-4
      dev.B2 = B2_2*1e-4
      amp = B1_2
      dev.tt = t_2
      dev.udet = alpha_2 # detuning energy     
      
      if S_ini == "|0+>":
        State2 = 0 
      elif S_ini == "|0->":
        State2 = 1
      elif S_ini == "|1+>":
        State2 = 2
      elif S_ini == "|1->":
        State2 = 3
      # indicate the processing status
      Rappture.Utils.progress(25,"Solving double qubit...")
      
      #main process
      dev.run(State2) #Quantum gate simulation                 
      Rappture.Utils.progress(90,"Solving double qubit...")

      #output the figures to the result of rappture 
      fig_TE2(dev.fig9,11,driver)
      fig = [ dev.fig10, dev.fig11,  dev.fig20, dev. fig21, dev.fig12]
      name = ['Density matrix (real part)','Density matrix (imag part)', 'Tomography (real)','Tomography (imag)','Truth table']
      
      for i in range(len(fig)):
        fig_put(fig[i],name[i],i+30,driver)              
          
      #define the output interface
      fig_rap1(dev.fig1,1,driver,amp)
#      fig_rap2(dev.fig2,2,driver)
#      fig_rap31(dev.fig31,31,driver)
#      fig_rap32(dev.fig32,32,driver)
#      fig_rap4(dev.fig4,4,driver)
#      fig_rap5(dev.fig5,5,driver)

      #bar3d(dev.data,driver)   # a simple code to generate the 3d tomograph figure roughly
      #tomo_rap(dev.fig6, driver) # draw plot version of DJ tomo
      #bloch_rap(dev.bloch_img_states, driver) # draw D-J bloch states

      Rappture.Utils.progress(99,"Logging...")
      
      # output log file
      label = ('II','IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ')#tick label for 2 qubit tomograph figure
      separator = '\n'+'='*70    #separator used in rappture output
      separator2 = '\n'+'-'*70   #separator2 used in rappture output
      
      #output contetnts: the input parameters and some of the result to a log file
      # it goes to the output text file of rappture
      output_2qubit=separator+'\n'+'Input parameters:\n\nU  = '+str(U)+'\nB1 = '+str(B1)+'\nB2 = '+str(B2)+'\nt  = '+str(t)+'\nT2 = '+str(T2)+'us\ndetuning energy = '+str(alpha)+'\n'+separator+'\nOutput data:\n\nFidelity = '+str(dev.Fidelity)+'\nGate delay = '+str(dev.delay)+'ns '+separator2+'\n\nThe output density matrix (real):\n'+str(np.real(dev.rof_qg))+'\n\nThe output density matrix (imag):\n'+str(np.imag(dev.rof_qg))+separator2+'\n\nTruthtable of 2 qubit gate:\n'+str(dev.rop_tt)+separator2+'\n\nTomography of 2 qubit gate:\n'+'Tick label for x- and y-axis:\n'+str(label)+'\nZdata:\n'+str(dev.Tm)+separator
      
      # push the output file to the difined library
      driver.put('output.log()',output_2qubit,append=1)
      driver.put('output.log().about.label','Quantum gate output data',append=0)
      
      Rappture.result(driver) # output the results 
      sys.exit() #exit

