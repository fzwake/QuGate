import numpy as np
import matplotlib.pyplot as plt
import scipy
from bloch import Bloch

def magnetic_field(state,x, y, z, h, s, t, phi):
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
    S = x * S_x + y * S_y + z * S_z + h * S_h + s * S_s + t * S_t

    # Hamiltonian for spinning particle at rest in magnetic field
    gamma = 1
    hbar = 1
    H = -gamma * hbar * S / 2

    # Solving shroedinger equation to get an evolution operator:
    def U(t):
        return scipy.linalg.expm(1j * H  * t )

    # Test state
    #state = 1/np.sqrt(5) *(2*np.array([1,0]) + np.array([0,1]))
    #state=np.array([0,1])
    # Will need to add decoherence later
    state_rho = np.outer(state, np.conj(state))

    # Do evolution
    t = np.arange(0, 2*np.pi, 0.1)
    U_t = [U(i) for i in t] 
    state_t_rho = [U.dot(state_rho).dot(U.conj().T) for U in U_t]

    return state_t_rho

    #esp_S_x = [state.conj().T.dot(S_x.dot(state)) for state in state_t]
    #esp_S_y = [state.conj().T.dot(S_y.dot(state)) for state in state_t]
    #esp_S_z = [state.conj().T.dot(S_z.dot(state)) for state in state_t]
    #plt.plot(esp_S_x, label="Expectation value of S_x")
    #plt.plot(esp_S_y, label="Expectation value of S_y")
    #plt.plot(esp_S_z, label="Expectation value of S_z")
    #plt.xlabel("t")
    #plt.ylabel("expectation")
    #plt.legend()
    #plt.show()
    #rho = np.outer(state, state)
    #print(rho)

#def dephasing(flag_gate):
#    S_x = np.array([[0,1], [1,0]])# x gate
#    S_y = np.array([[0,-1j], [1j,0]])# y gate
#    S_z = np.array([[1,0], [0,-1]])# z gate
#    if flag_gate==0:
#        S = S_x # x gate
#    elif flag_gate==1:
#        S = S_y # y gate
#    elif flag_gate==2:
#        S = S_z # z gate
def dephasing(state, x, y, z, h, s, r, phi):
    # Pauli matrices
    S_x = np.array([[0,1], [1,0]])
    S_y = np.array([[0,-1j], [1j,0]])
    S_z = np.array([[1,0], [0,-1]])
    #Hadamard 
    S_h = np.array([[1,1], [1,-1]])/np.sqrt(2)
    #S_gate
    S_s = np.array([[1,0], [0,1j]])
    #phase shift
    S_r = np.array([[1,0], [0,np.exp(1j*phi)]])
    # B.dot(S)
    S = x * S_x + y * S_y + z * S_z + h * S_h + s * S_s + r * S_r
    
    T = 1
    gamma = 1 / T
    hbar = 1
    # Hamiltonian
    H = hbar/2 * S

#    muB = 9.27e-24
#    B = 1.0
#    q = 1.6e-19
#    Ez=muB*2*B/q
#    H = Ez*S
    
    #state = 1/np.sqrt(5) *(2*np.array([1,0]) + np.array([0,1]))
#    state = np.array([1,0])
    rho = np.outer(state, np.conj(state))
#    print(rho)
#    b = Bloch()
#    b.plot_state(rho)
#    L = np.sqrt(gamma/2) * S_z
#    L_d = L.conj().T
#
#    drho_dt = -1j/hbar * (H.dot(rho) - rho.dot(H)) + 2 * L.dot(rho).dot(L_d) - L.dot(L_d.dot(rho)) + rho.dot(L.dot(L_d))
#

    tv = np.linspace(0, np.pi, 401)
    dt = tv[1] - tv[0]
    L = [np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]])]

    n = 0.1
    rho_t = []
    for i in range(len(tv-1)):
        drop = -1j * (H.dot(rho) - rho.dot(H))*dt  
        for ii_n in range(len(L)):
            temp = L[ii_n]
#            print(temp)
            drop=drop+dt*n*(temp.dot(rho).dot(temp.T)-1/2*(temp.T.dot(temp).dot(rho)+rho.dot(temp.T).dot(temp)))          
        rho = rho + drop
        rho_t.append(rho)
#        print(rho)
    rho_x=np.trace(S_x.dot(rho))
    rho_y=np.trace(S_y.dot(rho))
    rho_z=np.trace(S_z.dot(rho))
#    print(rho,rho_x, rho_y,rho_z)
    
    b = Bloch()
    for i in [0,100,200,300,400]:
        s=rho_t[i]
        b.plot_state(s)

    return rho_t, rho_x, rho_y,rho_z, S    

if __name__ == "__main__":
    # Apply magnet field in z direction
#    magnetic_field(0,0,1)
#    rho,rho_x, rho_y,rho_z=dephasing(2)
    rho,rho_x, rho_y,rho_z, S=dephasing([0,1],0,1,0,0,0,0,0)
    for i in [0,100,200,300,400]:
        b=Bloch()
        s=rho[i]
        b.plot_state(s)
        b.plot_state(S, con = -1, amp = 0.7, c='b')
        ax = b.fig.axes[0]
        ax.set_title('Rotation of spin from its initial state without decoherence', fontsize=2)
#        b.get_image_base64()
#        b.clear()
    b=Bloch()
    s=rho[1]
    b.plot_state(s)
    b.plot_state(S, c='b')
#    fig = b.fig
#    left, bottom, width, height = 0, 0, 0.3, 0.3
#    ax = fig.add_axes([left, bottom, width, height])
#    from bloch import Arrow3D
#    x = np.array([0, 1])
#    y = np.array([0, 1])
#    z = 3*np.array([0, 1])
#    a = Arrow3D(x, y, z, lw=2, arrowstyle='-|>',
#                mutation_scale=12, color='r')
#    ax.add_artist(a)
    


#b.fig.savefig('bloch.png',dpi=600,bbox_inches='tight')