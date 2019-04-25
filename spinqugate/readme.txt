
This folder contains the source files for GUI virsion of QuGate.
This is implemented by using rappture and python. The develope environment is python 3.6 in Spyder 3.2.8 on windows 10.
The main packages we use are:
    numpy         1.14.3
    matplotlib    2.2.2
    scipy         1.1.0

The Rappture-Python API is used for GUI. 
When you install the source code, specify the matlab path at line 30 of the fettoywr file.

File structure:
  tool.xml: Rappture file. Input GUI is set in this file.
  main.py: main python code. Get the input parameters from the Rappture input interface, and output the results to Rappture.
  
  Qugate.py: Double quantum gate class is defined.
  Qgate1.py: Single quantum gate class is defined.
  single_spin.py: Code for arbiraty rotation.
  figure.py: Some communication functions based on Rappture python API.  

================================================================================
Feb. 2019.

Tong Wu (twu1994@ufl.edu) and Jing Guo(guoj@ufl.edu)