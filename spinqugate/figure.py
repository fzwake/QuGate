import numpy as np
# function of generate each figure
def fig_rap1(fig,ind,driver,amp):
    # get figure data from the python figure 
    ind = 2 
    ax = fig.gca()
    lines = ax.lines
    xdata =  []
    xdata3 = []
    ydata = []
    ydata3 = []
    for i in range(len(lines)):
        xdata.append(lines[i].get_xdata())
        ydata.append(lines[i].get_ydata())
    
    ra = ydata[3][1]
    th1 = ra*1.3
    ra = ydata[4][1]
    th2 = ra*1.3
    for i in range(len(lines)):
        ydata2 = []
        xdata2 = []
        # set a certain region of y-axis to display for the best view        
        for i_data in range(len(ydata[i])):
            if ydata[i][i_data]>(th2) and ydata[i][i_data]<th1:
                ydata2.append(ydata[i][i_data])
                xdata2.append(xdata[i][i_data])
        xdata3.append(xdata2)
        ydata3.append(ydata2)
    xdata = xdata3
    ydata = ydata3       
    
    # define a figure group name
    Item = 'Double Qubit: Energy levels'   
    
    # define a curve type figure with a label
    driver.put('output.curve(f11).about.label','S11',append=0)
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    driver.put('output.curve(f11).xaxis.label','detunning gate voltage',append=0)
    driver.put('output.curve(f11).xaxis.units','mV',append=0)  
    driver.put('output.curve(f11).about.group',Item,append=0) # belong the figure to a certain group
    driver.put('output.curve(f11).about.style','-color blue -linestyle solid',append=0)
    driver.put('output.curve(f11).yaxis.label','Emp',append=0)
    driver.put('output.curve(f11).yaxis.units','meV',append=0)    

    for ii in range(len(xdata[0])):
        # output the figure data in rappture style 
        line = "%g %g\n" % (xdata[0][ii], ydata[0][ii])
        # push the data to a defined figure
        driver.put('output.curve(f11).component.xy', line, append=1)   
        
    driver.put('output.curve(f12).about.label','T0',append=0)
    driver.put('output.curve(f12).about.group',Item,append=0)
    driver.put('output.curve(f12).about.style','-color red -linestyle solid',append=0)
    for ii in range(len(xdata[1])):
        line = "%g %g\n" % (xdata[1][ii], ydata[1][ii])
        driver.put('output.curve(f12).component.xy', line, append=1)     
        
    driver.put('output.curve(f13).about.label','S20',append=0)
    driver.put('output.curve(f13).about.group',Item,append=0)
    driver.put('output.curve(f13).about.style','-color green -linestyle solid',append=0)
    for ii in range(len(xdata[2])):
        line = "%g %g\n" % (xdata[2][ii], ydata[2][ii])
        driver.put('output.curve(f13).component.xy', line, append=1)     
            
    driver.put('output.curve(f14).about.label','T1',append=0)
    driver.put('output.curve(f14).about.group',Item,append=0)
    driver.put('output.curve(f14).about.style','-color black -linestyle solid',append=0)
    for ii in range(len(xdata[3])):
        line = "%g %g\n" % (xdata[3][ii], ydata[3][ii])
        driver.put('output.curve(f14).component.xy', line, append=1)     

    driver.put('output.curve(f15).about.label','T-1',append=0)
    driver.put('output.curve(f15).about.group',Item,append=0)
    driver.put('output.curve(f15).about.style','-color yellow -linestyle solid',append=0)
    for ii in range(len(xdata[4])):
        line = "%g %g\n" % (xdata[4][ii], ydata[4][ii])
        driver.put('output.curve(f15).component.xy', line, append=1)
        
    driver.put('output.curve(f16).about.label','Operation point',append=0)
    driver.put('output.curve(f16).about.group',Item,append=0)
    driver.put('output.curve(f16).about.style','-color brown -linestyle dashed',append=0)
    for ii in range(len(xdata[5])):
        line = "%g %g\n" % (xdata[5][ii], ydata[5][ii])
        driver.put('output.curve(f16).component.xy', line, append=1)        
def fig_rap2(fig,ind,driver):
    ### Fig of mixing with S2D 
    ax = fig.gca()
    lines = ax.lines
    xdata =  []
    ydata = []
    for i in range(len(lines)):
        xdata.append(lines[i].get_xdata())
        ydata.append(lines[i].get_ydata())
        
    Item  = 'Double Qubit: Visualize probability mixing with S20'
    
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    driver.put('output.curve(f21).about.label','lowest',append=0)
    driver.put('output.curve(f21).xaxis.label','delta Vg',append=0)
    driver.put('output.curve(f21).xaxis.units','mV',append=0)  
    driver.put('output.curve(f21).about.group',Item,append=0)
    driver.put('output.curve(f21).about.style','-color blue -linestyle solid',append=0)
     
    driver.put('output.curve(f21).yaxis.label','Pde(%)',append=0)
    #driver.put('output.curve(f21).yaxis.units','eV',append=0)    
    
    for ii in range(len(xdata[0])):
        line = "%g %g\n" % (xdata[0][ii], ydata[0][ii])
        driver.put('output.curve(f21).component.xy', line, append=1)   
        
    driver.put('output.curve(f22).about.label','mixing with S20',append=0)
    driver.put('output.curve(f22).about.group',Item,append=0)
    driver.put('output.curve(f22).about.style','-color red -linestyle solid',append=0)
     
    driver.put('output.curve(f22).yaxis.label','Pde(%)',append=0)
    
    for ii in range(len(xdata[1])):
        line = "%g %g\n" % (xdata[1][ii], ydata[1][ii])
        driver.put('output.curve(f22).component.xy', line, append=1)    
         
def fig_rap31(fig,ind,driver):

    ax = fig.gca()
    lines = ax.lines
    xdata =  []
    ydata = []
    for i in range(len(lines)):
        xdata.append(lines[i].get_xdata())
        ydata.append(lines[i].get_ydata())
        
    Item = 'Double Qubit: Resonance frequency versus gate voltage'
    
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    driver.put('output.curve(f311).about.label','S11',append=0)
    driver.put('output.curve(f311).xaxis.label','delta Vg (mV)',append=0)
    driver.put('output.curve(f311).xaxis.units','mV',append=0)  
    driver.put('output.curve(f311).about.group',Item,append=0)
    driver.put('output.curve(f311).about.style','-color blue -linestyle solid',append=0)
     
    driver.put('output.curve(f311).yaxis.label','f',append=0)
    driver.put('output.curve(f311).yaxis.units','Hz',append=0)    
    
    for ii in range(len(xdata[0])):
        line = "%g %g\n" % (xdata[0][ii], ydata[0][ii])
        driver.put('output.curve(f311).component.xy', line, append=1)   
        
    driver.put('output.curve(f312).about.label','T0',append=0)
    driver.put('output.curve(f312).about.group',Item,append=0)
    driver.put('output.curve(f312).about.style','-color red -linestyle solid',append=0)
    
    for ii in range(len(xdata[1])):
        line = "%g %g\n" % (xdata[1][ii], ydata[1][ii])
        driver.put('output.curve(f312).component.xy', line, append=1)     
      
def fig_rap32(fig,ind,driver):

    ax = fig.gca()
    lines = ax.lines
    xdata =  []
    ydata = []
    for i in range(len(lines)):
        xdata.append(lines[i].get_xdata())
        ydata.append(lines[i].get_ydata())
        
    Item = 'Double Qubit: Probability density versus gate voltage (region of figure of resonance frequency) '
    
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    driver.put('output.curve(f321).about.label','line1',append=0)
    driver.put('output.curve(f321).xaxis.label','delta Vg (mV)',append=0)
    driver.put('output.curve(f321).xaxis.units','mV',append=0)  
    driver.put('output.curve(f321).about.group',Item,append=0)
    driver.put('output.curve(f321).about.style','-color blue -linestyle solid',append=0)
    driver.put('output.curve(f321).yaxis.label','Pde[%]',append=0)
    
    for ii in range(len(xdata[0])):
        line = "%g %g\n" % (xdata[0][ii], ydata[0][ii])
        driver.put('output.curve(f321).component.xy', line, append=1)   

def fig_rap4(fig,ind,driver):
    ### Fig of time evolution

    ind = 4 
    ax = fig.gca()
    lines = ax.lines
    xdata =  []
    ydata = []
    for i in range(len(lines)):
        xdata.append(lines[i].get_xdata())
        ydata.append(lines[i].get_ydata())
        

    Item = 'Double Qubit: Time Evolution (probability as time)'   
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    driver.put('output.curve(f41).about.label','00',append=0)
    driver.put('output.curve(f41).xaxis.label','time',append=0)
    driver.put('output.curve(f41).xaxis.units','s',append=0)  
    driver.put('output.curve(f41).about.group',Item,append=0)
    driver.put('output.curve(f41).about.style','-color blue -linestyle solid',append=0)
    driver.put('output.curve(f41).yaxis.label','Pde(%)',append=0)
    #driver.put('output.curve(f41).yaxis.units','%',append=0)    
    
    for ii in range(len(xdata[0])):
        line = "%g %g\n" % (xdata[0][ii], ydata[0][ii])
        driver.put('output.curve(f41).component.xy', line, append=1)   
        
    driver.put('output.curve(f42).about.label','01',append=0)
    driver.put('output.curve(f42).about.group',Item,append=0)
    driver.put('output.curve(f42).about.style','-color red -linestyle solid',append=0)
    for ii in range(len(xdata[1])):
        line = "%g %g\n" % (xdata[1][ii], ydata[1][ii])
        driver.put('output.curve(f42).component.xy', line, append=1)     
        
    driver.put('output.curve(f43).about.label','10',append=0)
    driver.put('output.curve(f43).about.group',Item,append=0)
    driver.put('output.curve(f43).about.style','-color green -linestyle dasedd',append=0)
    for ii in range(len(xdata[2])):
        line = "%g %g\n" % (xdata[2][ii], ydata[2][ii])
        driver.put('output.curve(f43).component.xy', line, append=1)     
            
    driver.put('output.curve(f44).about.label','11',append=0)
    driver.put('output.curve(f44).about.group',Item,append=0)
    driver.put('output.curve(f44).about.style','-color black -linestyle solid',append=0)
    for ii in range(len(xdata[3])):
        line = "%g %g\n" % (xdata[3][ii], ydata[3][ii])
        driver.put('output.curve(f44).component.xy', line, append=1)     
      
def fig_rap5(fig,ind,driver):

    ax = fig.gca()
    lines = ax.lines
    xdata =  []
    ydata = []
    for i in range(len(lines)):
        xdata.append(lines[i].get_xdata())
        ydata.append(lines[i].get_ydata())
        
    Item = 'Double Qubit: Angle of Rotation'
    
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    driver.put('output.curve(f51).about.label','line1 ',append=0)
    driver.put('output.curve(f51).xaxis.label','t',append=0)
    driver.put('output.curve(f51).xaxis.units','ns',append=0)  
    driver.put('output.curve(f51).about.group',Item,append=0)
    driver.put('output.curve(f51).about.style','-color blue -linestyle solid',append=0)
    
    driver.put('output.curve(f51).yaxis.label','phi(t) [/pi] ',append=0)
    
    for ii in range(len(xdata[0])):
        line = "%g %g\n" % (xdata[0][ii], ydata[0][ii])
        driver.put('output.curve(f51).component.xy', line, append=1)      
def fig_TE1(fig,ind,driver):
    ### Fig of time evolution

    ind = 4 
    ax = fig.gca()
    lines = ax.lines
    xdata =  []
    ydata = []
    for i in range(len(lines)):
        xdata.append(lines[i].get_xdata())
        ydata.append(lines[i].get_ydata())

    Item = 'Probability vs time (with dephasing effect)'   
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    driver.put('output.curve(f61).about.label','|0>',append=0)
    driver.put('output.curve(f61).xaxis.label','time',append=0)
    driver.put('output.curve(f61).xaxis.units','s',append=0)  
    driver.put('output.curve(f61).about.group',Item,append=0)
    driver.put('output.curve(f61).about.style','-color blue -linestyle solid',append=0)
    driver.put('output.curve(f61).yaxis.label','Pde(%)',append=0)
    #driver.put('output.curve(f41).yaxis.units','%',append=0)    
    
    for ii in range(len(xdata[0])):
        line = "%g %g\n" % (xdata[0][ii], ydata[0][ii])
        driver.put('output.curve(f61).component.xy', line, append=1)   
        
    driver.put('output.curve(f62).about.label','|1>',append=0)
    driver.put('output.curve(f62).about.group',Item,append=0)
    driver.put('output.curve(f62).about.style','-color red -linestyle solid',append=0)
    for ii in range(len(xdata[1])):
        line = "%g %g\n" % (xdata[1][ii], ydata[1][ii])
        driver.put('output.curve(f62).component.xy', line, append=1)     
        
    driver.put('output.curve(f65).about.label','pi rotation',append=0)
    driver.put('output.curve(f65).about.group',Item,append=0)
    driver.put('output.curve(f65).about.style','-color brown -linestyle dashed',append=0)
    for ii in range(len(xdata[2])):
        line = "%g %g\n" % (xdata[2][ii], ydata[2][ii])
        driver.put('output.curve(f65).component.xy', line, append=1)          

def fig_TE2(fig,ind,driver):
    ### Fig of time evolution

    ind = 4 
    ax = fig.gca()
    lines = ax.lines
    xdata =  []
    ydata = []
    for i in range(len(lines)):
        xdata.append(lines[i].get_xdata())
        ydata.append(lines[i].get_ydata())

    Item = 'Probability vs time (with dephasing effect)'   
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    driver.put('output.curve(f61).about.label','|0+>',append=0)
    driver.put('output.curve(f61).xaxis.label','time',append=0)
    driver.put('output.curve(f61).xaxis.units','s',append=0)  
    driver.put('output.curve(f61).about.group',Item,append=0)
    driver.put('output.curve(f61).about.style','-color blue -linestyle solid',append=0)
    driver.put('output.curve(f61).yaxis.label','Pde(%)',append=0)
    #driver.put('output.curve(f41).yaxis.units','%',append=0)    
    
    for ii in range(len(xdata[0])):
        line = "%g %g\n" % (xdata[0][ii], ydata[0][ii]*100)
        driver.put('output.curve(f61).component.xy', line, append=1)   
        
    driver.put('output.curve(f62).about.label','|0->',append=0)
    driver.put('output.curve(f62).about.group',Item,append=0)
    driver.put('output.curve(f62).about.style','-color red -linestyle solid',append=0)
    for ii in range(len(xdata[1])):
        line = "%g %g\n" % (xdata[1][ii], ydata[1][ii]*100)
        driver.put('output.curve(f62).component.xy', line, append=1)     
        
    driver.put('output.curve(f63).about.label','|1+>',append=0)
    driver.put('output.curve(f63).about.group',Item,append=0)
    driver.put('output.curve(f63).about.style','-color green -linestyle dasedd',append=0)
    for ii in range(len(xdata[2])):
        line = "%g %g\n" % (xdata[2][ii], ydata[2][ii]*100)
        driver.put('output.curve(f63).component.xy', line, append=1)     
            
    driver.put('output.curve(f64).about.label','|1->',append=0)
    driver.put('output.curve(f64).about.group',Item,append=0)
    driver.put('output.curve(f64).about.style','-color black -linestyle solid',append=0)
    for ii in range(len(xdata[3])):
        line = "%g %g\n" % (xdata[3][ii], ydata[3][ii]*100)
        driver.put('output.curve(f64).component.xy', line, append=1)  
        
    driver.put('output.curve(f65).about.label','pi rotation',append=0)
    driver.put('output.curve(f65).about.group',Item,append=0)
    driver.put('output.curve(f65).about.style','-color brown -linestyle dashed',append=0)
    for ii in range(len(xdata[4])):
        line = "%g %g\n" % (xdata[4][ii], ydata[4][ii])
        driver.put('output.curve(f65).component.xy', line, append=1)    
def bar3d(data,driver):

    xdata = data['x']
    ydata = data['y']
    zdata = data['z']    

    # create rappture mesh
    driver.put('output.mesh(m1).about.label','m1',append=0)
    driver.put('output.mesh(m1).dim',2,append=0)
    #driver.put('output.mesh(m1).units','um',append=0)
    driver.put('output.mesh(m1).hide','yes',append=0)
    driver.put('output.mesh(m1).grid.xaxis.min',0,append=0)
    driver.put('output.mesh(m1).grid.xaxis.max',4,append=0)
    driver.put('output.mesh(m1).grid.xaxis.numpoints','16',append=0)
    
    driver.put('output.mesh(m1).grid.yaxis.min','0',append=0)
    driver.put('output.mesh(m1).grid.yaxis.max','4',append=0)
    driver.put('output.mesh(m1).grid.yaxis.numpoints','16',append=0)
    Item  = 'Visualize probability mixing with S20'
    
    # Label the output graph with a title, x-axis label,
    # y-axis label, and y-axis units
    driver.put('output.field(fie1).about.label','Double Qubit: Deutsch-Jozsa algorithm tomography',append=0)
    driver.put('output.field(fie1).about.xaxis.label','Gate 1',append=0)
    driver.put('output.field(fie1).about.yaxis.label','Gate 2',append=0)
    driver.put('output.field(fie1).about.zaxis.label','real(rho)',append=0)
    driver.put('output.field(fie1).about.view','heightmap',append=0)
     
    driver.put('output.field(fie1).component.mesh','output.mesh(m1)', append=0)
    import numpy as np
    zdata = np.kron(zdata.reshape((4,4)),np.array([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]])).flatten() 
    zdata = ' '.join(map(str, zdata.ravel('F').tolist()))
    driver.put('output.field(fie1).component.values', zdata, append=0)   

def tomo_rap(image, driver):
    from io import BytesIO
    fig_io = BytesIO()
    image.savefig(fig_io, format='png')
    import base64
    figdata = base64.b64encode(fig_io.getvalue())
    driver.put('output.image(d0).about.label', 'Double Qubit: Deutsch-Jozsa algorithm tomography (img)', append=0)
    driver.put('output.image(d0).current', figdata, append=0)

def bloch_rap(images, driver):
    driver.put('output.sequence(bloch).about.label', 'Double Qubit: Bloch Sphere - Deutsh-Jozsa algorithm', append=0)
    driver.put('output.sequence(bloch).index.label', 'State', append=0)

    for idx, image in enumerate(images):
        driver.put('output.sequence(bloch).element({}).index'.format(idx), str(idx+1), append=0)
        driver.put('output.sequence(bloch).element({}).image.current'.format(idx), image, append=0)

def bloch_arbitrary_rotate(state, theta, phi, alpha, driver):
    driver.put('output.sequence(bloch_arbitrary_rotate).about.label', 'Single Qubit: Bloch Sphere - Arbitrary Rotation', append=0)
    driver.put('output.sequence(bloch_arbitrary_rotate).index.label', 'State', append=0)
    from bloch import Bloch
    import numpy as np
    b = Bloch()
    #state = np.array([1,0])
    state = np.outer(state, np.conj(state))
    b.plot_state(state)

    driver.put('output.sequence(bloch_arbitrary_rotate).element(0).index', str(1), append=0)
    driver.put('output.sequence(bloch_arbitrary_rotate).element(0).image.current', b.get_image_base64(), append=0)
    b.clear()

    axis = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    b.rotation_angle_axis(state, axis, alpha) 
    driver.put('output.sequence(bloch_arbitrary_rotate).element(1).index', str(2), append=0)
    driver.put('output.sequence(bloch_arbitrary_rotate).element(1).image.current', b.get_image_base64(), append=0)

def bloch_magnetic_field(state, magnetic_x, magnetic_y, magnetic_z, magnetic_h, magnetic_s, magnetic_r, magnetic_phi, driver):
    from single_spin import magnetic_field
    states = magnetic_field(state, magnetic_x, magnetic_y, magnetic_z,magnetic_h, magnetic_s, magnetic_r, magnetic_phi)
    from bloch import Bloch
    b = Bloch()
    for s in states:
        b.plot_state(s)

    driver.put('output.image(mag).about.label', 'Single Qubit: Bloch Sphere - Magnetic Field', append=0)
    driver.put('output.image(mag).current', b.get_image_base64(), append=0)

def bloch_time_evo2(rho, magnetic_x, magnetic_y, magnetic_z, magnetic_h, magnetic_s, magnetic_r, S, driver):
    driver.put('output.sequence(evo).about.label', 'Single Qubit: spin rotation', append=0)
    driver.put('output.sequence(evo).index.label', 'State', append=0)

    from bloch import Bloch
    import numpy as np
    idx = np.linspace(0,len(rho)-1,11)
    
    for i in range(len(idx)):
        b = Bloch()
        s = rho[int(idx[i])]
        b.plot_state(s)
        b.plot_state(S, con = -1, amp = 0.7, c='b')
        ax = b.fig.axes[0]
        ax.set_title('Spin rotation from initial state\n Red: Spin  Blue: B field', fontsize = 11)
        driver.put('output.sequence(evo).element('+str(i)+').index', str(i+1), append=0)
        driver.put('output.sequence(evo).element('+str(i)+').image.current', b.get_image_base64(), append=0)  

def fig_put(image, name, i, driver):
    from io import BytesIO
    fig_io = BytesIO()
    image.savefig(fig_io, format='png',bbox_inches ='tight')
    import base64
    figdata = base64.b64encode(fig_io.getvalue())
    driver.put('output.image('+str(i)+').about.label', name, append=0)
    driver.put('output.image('+str(i)+').current', figdata, append=0)
    