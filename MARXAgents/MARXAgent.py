import numpy as np
from scipy.linalg import inv, det
from numpy.linalg import slogdet
from scipy.stats import multivariate_normal
from scipy.special import gamma, gammaln
from scipy.optimize import minimize
import pickle
import os


class MARXAgent:
    """
    Active inference agent based on a Multivariate Auto-Regressive eXogenous model.

    Parameters are inferred through Bayesian filtering and controls through minimizing expected free energy.
    """

    def __init__(self, 
                 coefficients_mean_matrix, 
                 coefficients_row_covariance, 
                 precision_scale,
                 precision_degrees, 
                 control_prior_precision, 
                 goal_prior, 
                 Dy=2, 
                 Du=2,
                 delay_inp=1, 
                 delay_out=1, 
                 time_horizon=1, 
                 num_iters=10):
        
        self.Dy = Dy
        self.Dx = Du * (delay_inp+1) + Dy * delay_out
        self.Du = Du
        self.ybuffer = np.zeros((Dy, delay_out))
        self.ubuffer = np.zeros((Du, delay_inp+1))
        self.delay_inp = delay_inp
        self.delay_out = delay_out
        self.M = coefficients_mean_matrix
        self.Λ = coefficients_row_covariance
        self.Ω = precision_scale
        self.ν = precision_degrees
        self.Υ = control_prior_precision
        self.goal_prior = goal_prior
        self.thorizon = time_horizon
        self.num_iters = num_iters
        self.free_energy = float('inf')

    def update(self, y_k, u_k):
        
        M0 = self.M
        Λ0 = self.Λ
        Ω0 = self.Ω
        ν0 = self.ν

        self.ubuffer = self.backshift(self.ubuffer, u_k)
        x_k = np.concatenate([self.ubuffer.flatten(), self.ybuffer.flatten()])

        X = np.outer(x_k, x_k)
        Ξ = np.outer(x_k, y_k) + np.dot(Λ0, M0)

        self.ν = ν0 + 1
        self.Λ = Λ0 + X
        self.Ω = Ω0 + np.outer(y_k, y_k) + np.dot(M0.T, np.dot(Λ0, M0)) - np.dot(Ξ.T, np.dot(inv(Λ0 + X), Ξ))
        self.M = np.dot(inv(Λ0 + X), Ξ)

        self.ybuffer = self.backshift(self.ybuffer, y_k)

        # self.free_energy = -self.log_evidence(y_k, x_k)

        return None

    def params(self):
        return self.M, self.U, self.V, self.ν

    def log_evidence(self, y, x):
        η, μ, Ψ = self.posterior_predictive(x)
        return -0.5 * (self.Dy * np.log(η * np.pi) - np.log(det(Ψ)) - 2 * self.logmultigamma(self.Dy, (η + self.Dy) / 2) +
                       2 * self.logmultigamma(self.Dy, (η + self.Dy - 1) / 2) + (η + self.Dy) * np.log(1 + 1 / η * np.dot((y - μ).T, np.dot(Ψ, (y - μ)))))

    def posterior_predictive(self, x_t):
        η_t = self.ν - self.Dy + 1
        μ_t = np.dot(self.M.T, x_t)
        Ψ_t = (self.ν - self.Dy + 1) * inv(self.Ω)/(1 + np.dot(x_t, np.dot(inv(self.Λ), x_t)))
        return η_t, μ_t, Ψ_t

    def predictions(self, controls, time_horizon=1):
        m_y = np.zeros((self.Dy, time_horizon))
        S_y = np.zeros((self.Dy, self.Dy, time_horizon))

        ybuffer = self.ybuffer
        ubuffer = self.ubuffer

        for t in range(time_horizon):
            ubuffer = self.backshift(ubuffer, controls[:, t])
            x_t = np.concatenate([ubuffer.flatten(), ybuffer.flatten()])

            η_t, μ_t, Ψ_t = self.posterior_predictive(x_t)
            m_y[:, t] = μ_t
            S_y[:, :, t] = inv(Ψ_t) * η_t / (η_t - 2)

            ybuffer = self.backshift(ybuffer, m_y[:, t])

        return m_y, S_y

    def mutualinfo(self, x):
        _, _, Ψ = self.posterior_predictive(x)
        _,logdet = slogdet(Ψ)
        return logdet

    def crossentropy(self, x):
        m_star = self.goal_prior.mean
        S_star = self.goal_prior.cov
        η_t, μ_t, Ψ_t = self.posterior_predictive(x)
        return 0.5 * (η_t / (η_t - 2) * np.trace(np.dot(inv(S_star), inv(Ψ_t))) + np.dot((μ_t - m_star).T, np.dot(inv(S_star), (μ_t - m_star))))

    def EFE(self, controls):
        ybuffer = self.ybuffer
        ubuffer = self.ubuffer

        J = 0
        for t in range(self.thorizon):
            u_t = controls[t*self.Du:(t+1)*self.Du]
            ubuffer = self.backshift(ubuffer, u_t)
            x_t = np.concatenate([ubuffer.flatten(), ybuffer.flatten()])

            J += self.mutualinfo(x_t) + self.crossentropy(x_t) + np.dot(u_t, np.dot(self.Υ, u_t))/2.0

            _, m_y, _ = self.posterior_predictive(x_t)
            ybuffer = self.backshift(ybuffer, m_y)

        return J

    def minimizeEFE(self, u_0=None, time_limit=10, verbose=False, control_lims=(-np.inf, np.inf)):
        if u_0 is None:
            u_0 = 1e-8 * np.random.randn(self.thorizon)

        def J(u):
            return self.EFE(u)

        bounds = [control_lims] * u_0.size
        result = minimize(J, u_0, method='L-BFGS-B', bounds=bounds, options={'disp': verbose, 'maxiter': 10000})
        return result.x

    def backshift(self, x, a):
        if x.ndim == 2:
            return np.column_stack((a, x[:, :-1]))
        elif x.ndim == 1:
            N = x.size
            S = np.eye(N, k=-1)
            e = np.zeros(N)
            e[0] = 1.0
            return S.dot(x) + e * a

    def update_goals(self, x, g):
        x = np.roll(x, -1)
        x[-1] = g
        return x

    def multigamma(self, p, a):
        result = np.pi ** (p * (p - 1) / 4)
        for j in range(1, p + 1):
            result *= gamma(a + (1 - j) / 2)
        return result

    def logmultigamma(self, p, a):
        result = p * (p - 1) / 4 * np.log(np.pi)
        for j in range(1, p + 1):
            result += gammaln(a + (1 - j) / 2)
        return result

    def save_agent(self, filename, makedir=False):
        if not filename.endswith(".pkl"):
            print("Filename does not end with .pkl. Appending .pkl to filename")
            filename += ".pkl"

        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            if makedir:
                print(f"Directory {directory} does not exist. Creating directory")
                os.makedirs(directory)
            else:
                raise FileNotFoundError(f"Directory {directory} does not exist. Agent cannot be saved to {filename}")

        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Agent saved to {filename}")

    def load_agent(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError("File does not exist.")

        with open(filename, 'rb') as file:
            return pickle.load(file)

    def reset_buffer(self):
        self.ubuffer = np.zeros((self.Du, self.delay_inp))
        self.ybuffer = np.zeros((self.Dy, self.delay_out))
        self.free_energy = float('inf')


def acc2pos(acc, prev_state, dt=1.0):
    "Kalman filter for accelerometer integration"

    # State transition matrix
    A = np.array([[1, 0, 0, dt,  0,  0, dt**2/2,       0,       0],
                  [0, 1, 0,  0, dt,  0,       0, dt**2/2,       0],
                  [0, 0, 1,  0,  0, dt,       0,       0, dt**2/2],
                  [0, 0, 0,  1,  0,  0,      dt,       0,       0],
                  [0, 0, 0,  0,  1,  0,       0,      dt,       0],
                  [0, 0, 0,  0,  0,  1,       0,       0,      dt],
                  [0, 0, 0,  0,  0,  0,       1,       0,       0],
                  [0, 0, 0,  0,  0,  0,       0,       1,       0],
                  [0, 0, 0,  0,  0,  0,       0,       0,       1]])
    
    # Process noise covariance matrix
    σ   = 1e-1
    block1 = np.diag(np.repeat([dt**5/20], 3))
    block2 = np.diag(np.repeat([dt**4/8], 3))
    block3 = np.diag(np.repeat([dt**3/6], 3))
    block4 = np.diag(np.repeat([dt**2/2], 3))
    block5 = np.diag(np.repeat([dt], 3))
    Q = σ*np.block([[block1,block2,block3],
                    [block2,block3,block4],
                    [block3,block4,block5]])
    
    # Measurement matrix
    C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0]])
    
    # Measurement noise covariance matrix
    ρ = 1e-2
    R = np.diag(ρ*np.ones(3))

    # Prediction step
    state_pred_m = A @ prev_state.mean
    state_pred_S = A @ prev_state.cov @ A.T + Q

    # Correction step
    Is = C @ state_pred_S @ C.T + R
    Kg = state_pred_S @ C.T @ inv(Is)
    state_m = state_pred_m + Kg @ (acc - C @ state_pred_m)
    state_S = (np.eye(9) - Kg @ C) @ state_pred_S

    state = multivariate_normal(state_m,state_S)

    return state.mean[0:2], state