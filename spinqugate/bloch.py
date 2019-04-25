import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import scipy.linalg

# Hack to draw nice arrows
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)

        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)

        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Bloch(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(5, 5))
        self.axes = Axes3D(self.fig, azim=-60, elev=30)
        self.sphere_color = '#FFDDDD'
        self.sphere_alpha = 0.2
        self.frame_color = 'gray'
        self.frame_width = 1.5
        self.frame_alpha = 0.2
        self.draw_sphere()

    def draw_sphere(self):
        self.plot_front()
        self.plot_back()
        t = {'fontsize': 18, 'color': 'black',
             'horizontalalignment': 'center', 'verticalalignment': 'center'}
        self.axes.text(0, -1.2, 0, '$x$', **t)
        self.axes.text(1.2, 0, 0, '$y$', **t)
        self.axes.text(0, 0, 1.2, r'$\left|0\right>$', **t)
        self.axes.text(0, 0, -1.2, r'$\left|1\right>$', **t)
        self.plot_axes()
        # Hide axis
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_xaxis().set_ticks([])
        self.axes.get_yaxis().set_ticks([])
        self.axes.set_zticks([])
        self.axes.grid(False)
        self.axes._axis3don = False
        # Hide grid panes
        # self.axes.set_frame_on(False)
        self.axes.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.axes.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.axes.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    def plot_state(self, rho, con = 0, amp = 1, c='r'):
        # want x to be out of screen
        u, v, w = self.state_to_bloch(rho)
        x = v * np.array([con, 1])*amp
        y = -u * np.array([con, 1])*amp
        z = w * np.array([con, 1]) *amp    
        a = Arrow3D(x, y, z, lw=2, arrowstyle='-|>',
                    mutation_scale=12, color=c)
        self.axes.add_artist(a)

    def rotation_angle_axis(self, state, axis, angle):
        sigma = [np.array([[0, 1], [1,0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0,-1]])]
        nsigma = np.sum([a * s for a,s in zip(axis, sigma)], axis=0)
        r_n = scipy.linalg.expm(-1j * angle/2 * nsigma)
        for t in np.arange(0.01, angle/2, 0.1):
            r_t = scipy.linalg.expm(-1j * t * nsigma)
            prod = r_t.dot(state).dot(r_t.conj().T)
            prod = self.state_to_bloch(prod)
            self.axes.scatter(prod[1], -prod[0], prod[2], s=10,
                              alpha=1, edgecolor='none', zdir='z', color='r')

        self.plot_state(r_n.dot(state).dot(r_n.conj().T))
        #self.plot_evolution(state, r_n)



    def state_to_bloch(self, rho):
        # todo: use sigma{x,y,z}
        u = rho[1, 0] + rho[0, 1]
        #u = 2*np.real(rho[0,1])
        #v = 2*np.imag(rho[1,0])
        v = 1j*(rho[0, 1] - rho[1, 0])
        w = rho[0, 0] - rho[1, 1]
        u = np.real(u)
        v = np.real(v)
        w = np.real(w)
        return u, v, w

    def M(self, axis, theta):
        # inerpolate axis-angle rotation
        return scipy.linalg.expm(np.cross(np.eye(3), axis/scipy.linalg.norm(axis)*theta))

    def plot_trace(self, statei, statef, gate=None, c='r'):
        statei_vec = np.array(self.state_to_bloch(statei))
        statef_vec = np.array(self.state_to_bloch(statef))
        axisofrot = np.cross(statei_vec, statef_vec)
        angleofrot = np.arccos(np.dot(statei_vec, statef_vec))
        for i in np.arange(0, angleofrot, 0.1):
            M0 = self.M(axisofrot, i)
            prod = np.dot(M0, statei_vec)
            self.axes.scatter(prod[1], -prod[0], prod[2], s=10,
                              alpha=1, edgecolor='none', zdir='z', color=c)

    def get_and_plot_bloch_point(self, state_i, gate):
        '''
            Apply gate to state and then plot as point
        '''
        prod = gate.dot(state_i).dot(gate.conj().T) 
        B = self.state_to_bloch(p_trace_B(prod))
        A = self.state_to_bloch(p_trace_A(prod))
        self.axes.scatter(B[1], -B[0], B[2], s=10,
                          alpha=1, edgecolor='none', zdir='z', color='r')
        self.axes.scatter(A[1], -A[0], A[2], s=10,
                          alpha=1, edgecolor='none', zdir='z', color='b')

        return (np.array([B[1], -B[0], B[2]]), np.array([A[1], -A[0], A[2]]))

    def plot_point(self, point):
        self.axes.scatter(point[1], -point[0], point[2], s=10,
                          alpha=1, edgecolor='none', zdir='z', color='black')


    def fit_plane(self, XYZ):
        '''
            Fit plane using SVD
        '''
        mean = np.sum(XYZ, axis=0) / XYZ.shape[0]
        u,s,v = np.linalg.svd(XYZ - mean)
        normal = v.conj().T[:,-1];
        d = -mean.dot(normal)
        return (d, normal)

    def plot_plane(self, points, c = 'r'):
        '''
            Fits a plane to points and plots it
        '''
        points = np.array(points)
        d, normal = self.fit_plane(points)
        self.plot_expected_plane(normal, d, c=c)

    def plot_expected_plane(self, normal, d=0, c='r'):
        '''
            Plot a plane with some normal and d following
            ax + by + cz = d
        '''
        if normal[0] != 0:
            zz, yy = np.meshgrid(np.linspace(-1.0, 1.0, 2), np.linspace(-1.0, 1.0, 2))
            # ax+by+cz = d
            xx = (-normal[2]*zz - normal[1]*yy - d)*1. / normal[0]
            self.axes.plot_surface(xx, yy, zz, alpha=0.1, color=c)
        else:
            xx, yy = np.meshgrid(np.linspace(-1.0, 1.0, 2), np.linspace(-1.0, 1.0, 2))
            # ax+by+cz = d
            zz = (-normal[0]*xx - normal[1]*yy - d)*1. / normal[2]
            self.axes.plot_surface(xx, yy, zz, alpha=0.1, color=c)



    def plot_axes(self):
        # axes on bloch sphere
        span = np.linspace(-1.0, 1.0, 2)
        self.axes.plot(span, 0 * span, zs=0, zdir='z', label='X',
                       lw=1, color='gray')
        self.axes.plot(0 * span, span, zs=0, zdir='z', label='Y',
                       lw=1, color='gray')
        self.axes.plot(0 * span, span, zs=0, zdir='y', label='Z',
                       lw=1, color='gray')

    def plot_back(self):
        # back half of sphere
        u = np.linspace(0, np.pi, 25)
        v = np.linspace(0, np.pi, 25)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        self.axes.plot_surface(x, y, z, rstride=2, cstride=2,
                               color=self.sphere_color, linewidth=0,
                               alpha=self.sphere_alpha)
        # wireframe
        self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                                 color=self.frame_color,
                                 alpha=self.frame_alpha)
        # equator
        self.axes.plot(1.0 * np.cos(u), 1.0 * np.sin(u), zs=0, zdir='z',
                       lw=self.frame_width, color=self.frame_color)
        self.axes.plot(1.0 * np.cos(u), 1.0 * np.sin(u), zs=0, zdir='x',
                       lw=self.frame_width, color=self.frame_color)

    def plot_front(self):
        # front half of sphere
        u = np.linspace(-np.pi, 0, 25)
        v = np.linspace(0, np.pi, 25)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        self.axes.plot_surface(x, y, z, rstride=2, cstride=2,
                               color=self.sphere_color, linewidth=0,
                               alpha=self.sphere_alpha)
        # wireframe
        self.axes.plot_wireframe(x, y, z, rstride=5, cstride=5,
                                 color=self.frame_color,
                                 alpha=self.frame_alpha)
        # equator
        self.axes.plot(1.0 * np.cos(u), 1.0 * np.sin(u),
                       zs=0, zdir='z', lw=self.frame_width,
                       color=self.frame_color)
        self.axes.plot(1.0 * np.cos(u), 1.0 * np.sin(u),
                       zs=0, zdir='x', lw=self.frame_width,
                       color=self.frame_color)

    def clear(self):
        self.axes.clear()
        self.draw_sphere()

    def get_image_base64(self):
        # base64 representation for generated plot
        from io import BytesIO
        fig = BytesIO()
        self.fig.savefig(fig, format='png')
        import base64
        figdata = base64.b64encode(fig.getvalue())
        return figdata


def expm(A):
    d, Y = np.linalg.eig(A)
    Yinv = np.linalg.pinv(Y)
    D = np.diag(np.exp(d))
     
    Y = np.asmatrix(Y)
    D = np.asmatrix(D)
    Yinv = np.asmatrix(Yinv)
     
    B = Y*D*Yinv
    return B

def p_trace_B(rop):
    temp = np.zeros((2,2), dtype=complex)
    temp[0,0] = rop[0,0] + rop[1,1]
    temp[0,1] = rop[0,2] + rop[1,3]
    temp[1,0] = rop[2,0] + rop[3,1]
    temp[1,1] = rop[2,2] + rop[3,3]
    return temp
def p_trace_A(rop):
    temp = np.zeros((2,2), dtype=complex)
    temp[0,0] = rop[0,0] + rop[2,2]
    temp[0,1] = rop[0,1] + rop[2,3]
    temp[1,0] = rop[1,0] + rop[3,2]
    temp[1,1] = rop[1,1] + rop[3,3]
    return temp


def amp_damping(r, gamma, p):
    return np.array([r[0] * np.sqrt(1-gamma), r[1] * np.sqrt(1-gamma), gamma * (2*p -1) + r[2] * (1-gamma)])

def phase_flip(b, p):
    x = r*np.sin(theta)*np.cos(phi) * (1-2 * p)
    y = r*np.sin(theta)*np.sin(phi) * (1 - 2*p)
    z = -r*np.cos(theta)
    b.axes.plot_surface(x, y, z, color=b.sphere_color, alpha=b.sphere_alpha+0.2)
    z = r*np.cos(theta)
    b.axes.plot_surface(x, y, z, color=b.sphere_color, alpha=b.sphere_alpha+0.2)

def phase_damping(r, l):
    return np.array([r[0]*np.sqrt(1-l), r[1]*np.sqrt(1-l), r[2]])

if __name__ == "__main__":
    # test stuff
    b = Bloch()
    s = np.array([[1, 0], [0, 0]])

    n_theta = 10 # number of values for theta
    n_phi = 20  # number of values for phi
    r = 1        #radius of sphere
    
    theta, phi = np.mgrid[0.0:0.5*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]
    
    p = 0.3
    phase_flip(b, 0.3)

#    coords = np.dstack((x,y,z))
#    for coord in coords:
#        for vec in coord:
#            b.plot_point(phase_flip(vec, 0.3))
#            #b.plot_point(amp_damping(vec, 0.5, 0.8))
#
#    z = r*np.cos(theta)
#
#    coords = np.dstack((x,y,z))
#    for coord in coords:
#        for vec in coord:
#            b.plot_point(phase_flip(vec, 0.3))
#            #b.plot_point(amp_damping(vec, 0.5, 0.8))
#

    
    
    #b.plot_state(np.array([[1, 0], [0, 0]]))
    #b.rotation_angle_axis(s, np.array([1,0,1])/np.sqrt(2), np.pi)
    #b.plot_evolution(np.array([[1,0], [0, 0]]), np.array([[1,1],[1,-1]]))
    #b.plot_state(np.array([[0, 1], [0, 0]]))
    #b.plot_trace(np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]]))
    plt.show()
