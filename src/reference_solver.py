import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata


class HeatEquationFDM:
    def __init__(self, L_x=1.0, L_t=1.0, N_x=20, N_t=20, alpha=0.01):
        self.L_x = L_x
        self.L_t = L_t
        self.N_x = N_x
        self.N_t = N_t
        self.alpha = alpha
        self.dx = L_x / (N_x - 1)
        self.dt = L_t / (N_t - 1)
        self.u_exact = None
        
    def _index(self, i, j):
        return i + j * self.N_x
        
    def solve(self):
        x = np.linspace(0, self.L_x, self.N_x)
        u_initial = np.sin(np.pi * x)
        
        num_points = self.N_x * self.N_t
        A = lil_matrix((num_points, num_points))
        b = np.zeros(num_points)
        
        for j in range(self.N_t):
            for i in range(self.N_x):
                k = self._index(i, j)
                
                if i == 0 or i == self.N_x - 1:
                    A[k, k] = 1
                    b[k] = 0
                elif j == 0:
                    A[k, k] = 1
                    b[k] = u_initial[i]
                elif j == self.N_t - 1:
                    A[k, k] = 1
                    A[k, self._index(i, j - 1)] = -1
                    b[k] = 0
                else:
                    A[k, k] = 1 + 2 * self.alpha * self.dt / self.dx**2
                    A[k, self._index(i - 1, j)] = -self.alpha * self.dt / self.dx**2
                    A[k, self._index(i + 1, j)] = -self.alpha * self.dt / self.dx**2
                    A[k, self._index(i, j - 1)] = -1
        
        self.u_exact = spsolve(A.tocsr(), b).reshape((self.N_t, self.N_x))
        return self.u_exact
    
    def interpolate(self, test_points):
        if self.u_exact is None:
            self.solve()
            
        x_grid, t_grid = np.meshgrid(
            np.linspace(0, self.L_x, self.N_x),
            np.linspace(0, self.L_t, self.N_t)
        )
        exact_points = np.column_stack((x_grid.flatten(), t_grid.flatten()))
        exact_values = self.u_exact.flatten()
        
        if hasattr(test_points, 'numpy'):
            test_points = test_points.numpy()
            
        return griddata(exact_points, exact_values, test_points, method='cubic')
