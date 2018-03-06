import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

class Support(object):
    """A support which represents an endpoint of a beam/beams.
    """

    def __init__(self, id, type, X, Z):
        """
        Attributes:
        id. number: A number which identifies a support.
        type: pinned, fixed, roller
        """
        self.id = id
        self.type = type
        self.X = X
        self.Z = Z
        self.Mat = {}
        self.u = None
        self.w = None
        self.Fx = None
        self.Fy = None
        self.M = None
        self.fi = None

    def set_boundrycondit(self, beam_id, beam_boundry):
        """Defines boundry conditions inflicted by support"""

        if self.type == "pinned":
            self.Mat[beam_id] = beam_boundry
            self.Main_boundry = ('u', 'w', 'M')
            self.Secondary_boundry = ()
            self.Konsistency = ()
            self.Irrelevant = ('Fx', 'Fy', 'fi')

        if self.type == "fixed":
            self.Mat[beam_id] = beam_boundry
            self.Main_boundry = ('u', 'w', 'fi')
            self.Secondary_boundry = ()
            self.Konsistency = ()
            self.Irrelevant = ('Fx', 'Fy', 'M')

        if self.type == "roller":
            self.Mat[beam_id] = beam_boundry
            self.Main_boundry = ('w', 'M')
            self.Secondary_boundry = ('Fx',)
            self.Konsistency = ('u',)
            self.Irrelevant = ('Fy', 'fi')

        if self.type == "joint":
            self.Mat[beam_id] = beam_boundry
            self.Main_boundry = ('M',)
            self.Secondary_boundry = ('Fx', 'Fy')
            self.Konsistency = ('u', 'w')
            self.Irrelevant = ('fi',)

        if self.type == "endpoint":
            self.Mat[beam_id] = beam_boundry
            self.Main_boundry = ()
            self.Secondary_boundry = ('Fx', 'Fy', 'M')
            self.Konsistency = ('u', 'w', 'fi')
            self.Irrelevant = ()

    def set_boundryforces(self, u, w, fi, Fx, Fy, M):
        """Sets deflection, forces and/or moment acting on the support."""
        self.u = u
        self.w = w
        self.fi = fi
        self.Fx = Fx
        self.Fy = Fy
        self.M = M

class Beam(object):
    """A beam which represents bearing element supported by supports. Boundry conditions are enforced by supportes.
    Beam has 2 supports.
    """

    def __init__(self, id, supports_1, supports_2, I, E, A, L, num_ele = 3):
        """
        Attributes:
            id. number: A number which identifies a beam
            support id: Tuple which appoints supports
            I: Second moment of area
            E: Elasticity module
            A: Cross section area
            num_ele: number of differential equations solved
        """
        self.id = id
        self.supports_1 = supports_1
        self.supports_2 = supports_2
        self.u = 0
        self.w = 0
        self.I = I
        self.L = L
        self.E = E
        self.A = A
        self.num_ele = num_ele
        self.h = self.L / (self.num_ele-1)

    def set_equations_u(self):
        """Constructs matrix containing general differential equation for deflection of the beam."""
        self.M_u = np.zeros((self.num_ele, self.num_ele+2))
        for i in range(self.num_ele):
            self.M_u[i, i:i+3] = np.array([1, -2, 1])*self.E*self.A/self.h**2

    def set_equations_w(self):
        """Constructs matrix containing general differential equation for tension of the beam."""
        self.M_w = np.zeros((self.num_ele, self.num_ele+4))
        for i in range(self.num_ele):
            self.M_w[i, i:i+5] = np.array([1, -4, 6, -4, 1]) *self.E*self.I/self.h**4

    def set_boundry_1(self):
        """Constructs matrix containing all boundry equations for first support."""
        self.B_1 = np.zeros((6, self.num_ele*2+6))
        #u
        self.B_1[0, 1] = self.tran_m[0, 0]
        self.B_1[0, self.num_ele+4] = self.tran_m[0, 1]
        #w
        self.B_1[1, 1] = self.tran_m[1, 0]
        self.B_1[1, self.num_ele+4] = self.tran_m[1, 1]
        #Fx
        self.B_1[2, 0:3] = np.array([-1, 0, 1])*self.E*self.A/(2*self.h)*self.tran_m[0, 0]
        self.B_1[2, self.num_ele+2:self.num_ele+7] = np.array([1, -2, 0, 2, -1])/(2*self.h**3)*self.E*self.I*self.tran_m[0, 1]
        #Fy
        self.B_1[3, 0:3] = np.array([-1, 0, 1])*self.E*self.A/(2*self.h)*self.tran_m[1, 0]
        self.B_1[3, self.num_ele+2:self.num_ele+7] = np.array([1, -2, 0, 2, -1])/(2*self.h**3)*self.E*self.I*self.tran_m[1, 1]
        #M
        self.B_1[4, self.num_ele+3:self.num_ele+6] = np.array([1, -2, 1])/(self.h**2)*self.E*self.I
        #fi
        self.B_1[5, self.num_ele+3:self.num_ele+6] = np.array([-1, 0, 1])/(2*self.h)

    def set_boundry_2(self):
        """Constructs matrix containing all boundry equations for second support."""
        self.B_2 = np.zeros((6, self.num_ele*2+6))
        #u
        self.B_2[0, self.num_ele] = self.tran_m[0, 0]
        self.B_2[0, -3] = self.tran_m[0, 1]
        #w
        self.B_2[1, self.num_ele] = self.tran_m[1, 0]
        self.B_2[1, -3] = self.tran_m[1, 1]
        #Fx
        self.B_2[2, self.num_ele-1:self.num_ele+2] = np.array([-1, 0, 1])*self.E*self.A/(2*self.h)*self.tran_m[0, 0]
        self.B_2[2, -5::] = np.array([1, -2, 0, 2, -1])/(2*self.h**3)*self.E*self.I*self.tran_m[0, 1]
        #Fy
        self.B_2[3, self.num_ele-1:self.num_ele+2] = -np.array([-1, 0, 1])*self.E*self.A/(2*self.h)*self.tran_m[1, 0]
        self.B_2[3, -5::] = np.array([1, -2, 0, 2, -1])/(2*self.h**3)*self.E*self.I*self.tran_m[1, 1]
        #M
        self.B_2[4, -4:-1] = np.array([1, -2, 1])/self.h**2*self.E*self.I
        #fi
        self.B_2[5, -4:-1] = np.array([-1, 0, 1])/(2*self.h)

    def form_M(self, supports_1, supports_2):
        """Forms final beam matrix."""

        M_ = block_diag(self.M_u, self.M_w)

        Naming1 = []
        Naming2 = []

        for sup in supports_1:
            for i in  ['u', 'w', 'Fx', 'Fy', 'M', 'fi']:
                Naming1.append([self.id, 1, sup.id, i])

        for sup in supports_2:
           for i in  ['u', 'w', 'Fx', 'Fy', 'M', 'fi']:
                 Naming2.append([self.id, 2, sup.id, i])

        Namingdif = [[self.id, None, None, i] for i in range(2*self.num_ele)]
        self.Naming = np.vstack((Naming1, Naming2, Namingdif))
        M_support_1 = np.vstack([sup.Mat[self.id] for sup in supports_1])
        M_support_2 = np.vstack([sup.Mat[self.id] for sup in supports_2])
        self.M = np.vstack((M_support_1, M_support_2, M_))

    def form_b(self, supports_1, supports_2):
        """Forms loading vector."""

        b = np.zeros(2*self.num_ele+6*(len(supports_1) + len(supports_2)))
        for num, sup in enumerate(supports_1):
            b[(num)*6:(num)*6+6] = np.array([sup.u, sup.w, sup.Fx, sup.Fy, sup.M, sup.fi])
        for num, sup in enumerate(supports_2):
            b[(num+len(supports_1))*6:(num+len(supports_1))*6+6] = np.array([sup.u, sup.w, sup.Fx, sup.Fy, sup.M, sup.fi])
        b[6*(len(supports_1)+len(supports_2)):6*(len(supports_1)+len(supports_2))+self.num_ele] = self.n
        b[6*(len(supports_1)+len(supports_2))+self.num_ele::] = self.q
        self.b = b

    def set_boundry_M(self, bound_matrices, sup):
        """Forms boundry condition matrix"""
        sup.set_boundrycondit(self.id, bound_matrices)

    def set_loading(self, n, q):
        """Sets loading values for beam.
        :param beam_matrices: tension loading, either function or descrete value/s
        :param q: shear loading, either function or descrete value/s
        """
        x = np.linspace(0, self.L, self.num_ele, endpoint=True)
        if callable(n):
            self.n = n(x)
        else:
            self.n = n
        if callable(q):
            self.q = q(x)
        else:
            self.q = q

    def transformation_matrix(self):
        """Forms and returns transformation matrix from local beam coorinates to global coordinates.
        """
        self.dX = self.supports_2[0].X - self.supports_1[0].X
        self.dZ = self.supports_2[0].Z - self.supports_1[0].Z
        self.tran_m = np.array([[self.dX / (np.sqrt(self.dX ** 2 + self.dZ ** 2)), self.dZ / (np.sqrt(self.dX ** 2 + self.dZ ** 2))],
                           [-self.dZ / (np.sqrt(self.dX ** 2 + self.dZ ** 2)), self.dX / (np.sqrt(self.dX ** 2 + self.dZ ** 2))]
                           ])

    def calculate_NTM(self):
        """Calculats descrete values of tension force, shear force, bending moment"""
        self.N = np.array([self.E*self.A*(self.u[i+1] - self.u[i-1])/(2*self.h) for i in np.arange(1, self.num_ele+1)])
        self.T = -np.array([self.E*self.I*(self.w[i+2] - 2*self.w[i+1] + 2*self.w[i-1] - self.w[i-2])/(2*self.h**3) for i in np.arange(1, self.num_ele+1)])
        self.M = np.array([self.E*self.I*(self.w[i+1] - 2*self.w[i] +self.w[i-1])/self.h**2 for i in np.arange(1, self.num_ele+1)])

    def drawing(self):
        """Drawing diagrams.
        u = elongation
        w = deflection
        T = Shear
        N = Tension force
        M = Bending moment
        """

        x = np.linspace(0, self.L, self.num_ele, endpoint=True)

        plt.figure(0)
        plt.plot(x, self.w[2:-2])
        plt.plot(0, 0, '.')
        plt.title('w')

        plt.figure(1)
        plt.plot(x, self.u[1:-1])
        plt.plot(0, 0, '.')
        plt.title('u')

        plt.figure(2)
        plt.plot(x, self.N)
        plt.plot(0, 0, '.')
        plt.title('N')

        plt.figure(3)
        plt.plot(x[1:], self.T[1:])
        plt.plot(0, 0, '.')
        plt.title('T')

        plt.figure(4)
        plt.plot(x, self.M)
        plt.title('M')
        plt.plot(0, 0, '.')

        plt.show()

def main_matrix_formation(beam_matrices, beam_loading, naming, supports):
    """
    Constructs the main matrix M and vector b needed for calculation. Also returns vector with identifications for rows in M.
    :param beam_matrices: Matrices of individual beams
    :param beam_loading: Loading vectors of individual beams
    :param naming: Naming vector of individual beam
    :param supports: Supports included in the system
    :return: M, b, Naming
    """

    M = block_diag(*beam_matrices)
    print(len(M[0]))
    b = np.hstack(beam_loading)
    Naming = np.vstack(naming)
    for Sup in supports:
        if len(Sup.Mat) > 1:
            for secondary_var, consistency_var in zip(Sup.Secondary_boundry, Sup.Konsistency):
                indices_secondary = [i for i, x in enumerate(Naming) if str(Sup.id) == x[2] and secondary_var == x[3]]
                indices_consistency = [i for i, x in enumerate(Naming) if str(Sup.id) == x[2] and consistency_var == x[3]]
                for index in indices_secondary[:-1]:
                    M[indices_secondary[-1]] = M[indices_secondary[-1]] + M[index]
                for num, index in enumerate(indices_consistency[1:]):
                    M[indices_consistency[num]] += M[index]
                delete = np.hstack((indices_consistency[-1], indices_secondary[:-1]))
                M = np.delete(M, delete, 0)
                b = np.delete(b, delete, 0)
                Naming = np.delete(Naming, delete, 0)
        if len(Sup.Mat) == 1:
            for consistency_var in Sup.Konsistency:
                indices_consistency = [i for i, x in enumerate(Naming) if str(Sup.id) == x[2] and consistency_var == x[3]]
                M = np.delete(M, indices_consistency, 0)
                b = np.delete(b, indices_consistency, 0)
                Naming = np.delete(Naming, indices_consistency, 0)
        for irrelevant_var in Sup.Irrelevant:
            indices_irrelevant = [i for i, x in enumerate(Naming) if str(Sup.id) == x[2] and irrelevant_var == x[3]]
            M = np.delete(M, indices_irrelevant, 0)
            b = np.delete(b, indices_irrelevant, 0)
            Naming = np.delete(Naming, indices_irrelevant, 0)

    return M, b, Naming

def resoult_parsing(r, *Beams):
    """Function for parsing the final resoult obtained via numpy."""
    x = 0
    for beam in Beams:
        beam.u = r[x:x+beam.num_ele+2]
        beam.w = r[x+beam.num_ele+2:x+beam.num_ele*2+6]
        x = x + beam.num_ele*2+6

def drawing(beams, supports):
    """Function for construcing a matplotlib plot for general shape of the system."""
    for beam in beams:
        plt.plot([beam.support_1.X, beam.support_2.X], [beam.support_1.Z, beam.support_2.Z], label=beam.id)
    for support in supports:
        plt.plot(support.X, support.Z, '*', label=support.id)
    plt.axis('equal')
    plt.legend()