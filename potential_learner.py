
import sys
import time
import h5py
import copy
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy import stats
from types import SimpleNamespace
from matplotlib import gridspec
from mpl_toolkits import mplot3d
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
sys.path.append('../')
from tools.bayes_tools import HistoryH5


# Units are nanoseconds (ns), nanometers (nm), attograms (ag)
PARAMETERS = {
    # Variables
    'P': None,  # probability
    'z': None,  # friction
    'v': None,  # measurement noise variance
    'f_data': None,  # force at data points
    'u_indu': None,  # potential at inducing points
    'u_grid': None,  # potential at grid points
    'x_data': None,  # positions at data time levels
    'x_grid': None,  # positions of grid points
    'x_indu': None,  # positions of inducing points
    'traj_mask': None,  # index of trajectories

    # Experiment
    'dt': 1,         # (ns)           chosen time step for Langevin dynamics
    'kT': 4.114,     # (ag*nm^2/ns^2) experiment temperature

    # Priors
    'x0_mean': None,  # hyperparameter
    'x0_vars': None,  # hyperparameter
    'z_shape': 2,     # hyperparameter
    'z_scale': None,  # hyperparameter
    'v_shape': 2,     # hyperparameter
    'v_scale': None,  # hyperparameter

    # Covariance matrix
    'sig': 10,           # hyperparameter for potential
    'ell': None,         # hyperparameter for potential
    'eps': .1,           # numerical stability parameter for matrix inversion
    'K_grid_indu': None,        # covariances between potential at grid and potential at indu
    'K_indu_indu': None,        # covariances between potential at indu and potential at indu
    'K_indu_indu_inv': None,
    'K_indu_indu_chol': None,

    # precomputed matrixes when v==0
    'K_data_indu': None,
    'K_tilde': None,
    'K_tilde_chol': None,

    # ground truth parameters
    'force': None,
    'potential': None,

    # Num
    'num_data': None,  # number of time levels
    'num_dims': None,  # number of data dimensions
    'num_traj': None,  # number of trajectories
    'num_indu': None,  # number of inducing points
    'num_grid': None,  # number of grid points

    # Sampler parameters
    'seed': 0,          # random number generator seed
    'x_prop_shape': 100,      # proposal distribution shape for friction
    'x_prop_std': None,     # proposal distribution width (STD) for position
    'parallelize': False,
}


class PotentialLearner:

    @ staticmethod
    def simulate_data(parameters=None, **kwargs):

        default_parameters = {
            'dt': 1,               # (ns)           chosen time step for Langevin dynamics
            'kT': 4.114,           # (ag*nm^2/ns^2) experiment temperature
            'z': 10000,            # friction
            'v': 10,               # measurement noise variance
            'x_data': None,        # trajectory
            'num_data': 10000,     # number of time levels
            'num_data_per': None,  # number of time levels per trajectory
            'num_dims': 1,         # number of data dimensions
            'num_traj': 1,         # number of trajectories
            'force': lambda x_: -25e-3 * x_,
            'seed': 0,
        }

        # set parameters
        if parameters is None:
            parameters = {**default_parameters, **kwargs}
        else:
            parameters = {**default_parameters, **parameters, **kwargs}
        dt = parameters['dt']
        kT = parameters['kT']
        z = parameters['z']
        v = parameters['v']
        f = parameters['force']
        num_data = parameters['num_data']
        num_data_per = parameters['num_data_per']
        num_dims = parameters['num_dims']
        num_traj = parameters['num_traj']
        seed = parameters['seed']

        # set rng
        np.random.seed(seed)

        # calculate values
        kick = 2 * dt * kT / z
        if num_data_per is None:
            num_data_per = num_data
        else:
            num_data = num_data_per * num_traj
        traj_mask = np.repeat(np.arange(num_traj), num_data_per)

        # sample trajectory
        x_data = np.zeros((num_data, num_dims))
        for t in range(num_traj):
            ids = np.where(traj_mask == t)[0]
            x_data_t = np.zeros((len(ids), num_dims))
            x_data_t[0, :] = stats.norm.rvs(loc=np.zeros(num_dims), scale=np.sqrt(kick))
            for n in range(1, num_data_per):
                x_data_t[n, :] = (
                    x_data_t[n - 1, :] 
                    + dt / z * f(x_data_t[n - 1, :]) 
                    + np.sqrt(kick) * np.random.randn(num_dims)
                )
            x_data[ids, :] = x_data_t

        # sample data
        if v > 0:
            data = stats.norm.rvs(loc=x_data, scale=np.sqrt(v))
        else:
            data = x_data.copy()

        # set parameters
        parameters['x_data'] = x_data
        parameters['num_data'] = num_data
        parameters['traj_mask'] = traj_mask

        return data, parameters

    @ staticmethod
    def initialize_variables(data, parameters, **kwargs) -> SimpleNamespace:

        # set up variables
        variables = SimpleNamespace(**{**PARAMETERS, **parameters, **kwargs})

        # extract variables
        v = variables.v
        z = variables.z
        dt = variables.dt
        kT = variables.kT
        sig = variables.sig
        eps = variables.eps
        ell = variables.ell
        x0_mean = variables.x0_mean
        x0_vars = variables.x0_vars
        v_shape = variables.v_shape
        v_scale = variables.v_scale
        z_shape = variables.z_shape
        z_scale = variables.z_scale
        x_prop_std = variables.x_prop_std
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_indu = variables.num_indu
        num_grid = variables.num_grid
        num_traj = variables.num_traj
        traj_mask = variables.traj_mask

        # Set up numbers
        num_data, num_dims = data.shape
        if num_indu is None:
            num_indu = [100, 25, 10][num_dims - 1]
        if num_grid is None:
            num_grid = [200, 50, 20][num_dims - 1]
        variables.num_data = num_data
        variables.num_dims = num_dims
        variables.num_indu = num_indu
        variables.num_grid = num_grid

        # set up trajectory mask
        if traj_mask is None:
            traj_mask = np.zeros(num_data, dtype=int)
        num_traj = int(np.max(traj_mask)) + 1
        variables.traj_mask = traj_mask
        variables.num_traj = num_traj

        # set noise
        if v is None:
            v = np.var(data) / 10
        if v_scale is None:
            v_scale = v / v_shape
        variables.v = v
        variables.v_scale = v_scale

        # set up friction
        if z is None:
            z = 2 * kT * dt / np.mean(np.sum((data[1:, :] - data[:-1, :]) ** 2, axis=1))
        if z_scale is None:
            z_scale = z / z_shape
        variables.z = z
        variables.z_scale = z_scale

        # Probability
        P = - np.inf
        variables.P = P

        # set up inducing points
        f_data = np.zeros((num_data, num_dims))
        u_indu = np.zeros((num_indu ** num_dims, 1))
        u_grid = np.zeros((num_grid ** num_dims, 1))
        variables.f_data = f_data
        variables.u_indu = u_indu
        variables.u_grid = u_grid

        # set up trajectory
        if x_prop_std is None:
            x_prop_std = .01 * (np.max(data) - np.min(data))
        if x0_mean is None:
            x0_mean = np.zeros(num_dims)
        if x0_vars is None:
            x0_vars = .5 * np.max(np.max(data, 0) - np.min(data, 0))
        if v == 0:
            x_data = data.copy()
        else:
            x_data = stats.norm.rvs(loc=data, scale=np.sqrt(v))
        x_indu = np.zeros((num_indu ** num_dims, num_dims))
        x_grid = np.zeros((num_grid ** num_dims, num_dims))
        for d in range(num_dims):
            temp_indu = .1 * (np.max(data) - np.min(data))
            temp_grid = .1 * (np.max(data) - np.min(data))
            temp_indu = np.linspace(np.min(data) - temp_indu, np.max(data) + temp_indu, num_indu)
            temp_grid = np.linspace(np.min(data) - temp_grid, np.max(data) + temp_grid, num_grid)
            x_indu[:, d] = np.tile(np.repeat(temp_indu, num_indu ** (num_dims - d - 1)), num_indu ** d)
            x_grid[:, d] = np.tile(np.repeat(temp_grid, num_grid ** (num_dims - d - 1)), num_grid ** d)
        variables.x_prop_std = x_prop_std
        variables.x0_mean = x0_mean
        variables.x0_vars = x0_vars
        variables.x_data = x_data
        variables.x_indu = x_indu
        variables.x_grid = x_grid

        # set up kernel
        if ell is None: # set length scale
            ell = .05 * (np.max(np.max(data, axis=0) - np.min(data, axis=0)))
        @nb.njit(cache=True)
        def kernel(x1, x2, d_dx=False, dd_ddx=False):

            sig2 = sig ** 2
            ell2 = ell ** 2
            ell4 = ell ** 4
            num_1, num_dims = x1.shape
            num_2, ________ = x2.shape

            if not (d_dx or dd_ddx):
                K = np.zeros((num_1, num_2))
                for i in range(num_1):
                    for j in range(num_2):
                        K[i, j] = sig2 * np.exp(-.5 * np.sum((x1[i, :] - x2[j, :]) ** 2) / ell2)
            else:
                K = np.zeros((num_1 * num_dims, num_2))
                for i in range(num_1):
                    for j in range(num_2):
                        Kij = sig2 * np.exp(-.5 * np.sum((x1[i, :] - x2[j, :]) ** 2) / ell2)
                        for d in range(num_dims):
                            if d_dx:
                                K[i + num_1 * d, j] = - Kij * (x1[i, d] - x2[j, d]) / ell2
                            elif dd_ddx:
                                K[i + num_1 * d, j] = - Kij * (ell2 - (x1[i, d] - x2[j, d]) ** 2) / ell4

            return K

        # set up covariance matrices
        K_grid_indu = kernel(x_grid, x_indu)
        K_indu_indu = kernel(x_indu, x_indu)
        K_indu_indu_inv = np.linalg.inv(K_indu_indu + eps * np.max(K_indu_indu) * np.eye(num_indu ** num_dims))
        K_indu_indu_chol = np.linalg.cholesky(K_indu_indu + eps * np.max(K_indu_indu) * np.eye(num_indu ** num_dims))
        variables.ell = ell
        variables.kernel = kernel
        variables.K_grid_indu = K_grid_indu
        variables.K_indu_indu = K_indu_indu
        variables.K_indu_indu_inv = K_indu_indu_inv
        variables.K_indu_indu_chol = K_indu_indu_chol

        # if there is no noise then K matrices can be pre-computed
        if v == 0:
            ids = np.where(traj_mask[1:] == traj_mask[:-1])[0]
            K_data_indu = - kT * kernel(x_data[ids, :], x_indu, d_dx=True)
            K_tilde = np.linalg.inv(K_indu_indu_inv + dt / (2 * z * kT) * K_indu_indu_inv @ K_data_indu.T @ K_data_indu @ K_indu_indu_inv)
            K_tilde_chol = np.linalg.cholesky(K_tilde + eps * np.eye(num_indu ** num_dims))
            variables.K_data_indu = K_data_indu
            variables.K_tilde = K_tilde
            variables.K_tilde_chol = K_tilde_chol

        return variables

    @staticmethod
    def sample_potential(data, variables):

        dt = variables.dt
        kT = variables.kT
        z = variables.z
        v = variables.v
        eps = variables.eps
        x_data = variables.x_data
        x_indu = variables.x_indu
        kernel = variables.kernel
        K_grid_indu = variables.K_grid_indu
        K_indu_indu_inv = variables.K_indu_indu_inv
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_indu = variables.num_indu
        num_traj = variables.num_traj
        num_grid = variables.num_grid
        traj_mask = variables.traj_mask

        # Calculate displacements
        ids = np.where(traj_mask[1:] == traj_mask[:-1])[0]
        y_data = x_data[1:, :] - x_data[:-1, :]
        y_data = y_data[ids, :].reshape((-1, 1), order='F')

        # Calculate covariance
        if v > 0:
            K_data_indu = - kT * kernel(x_data[ids, :], x_indu, d_dx=True)
            K_tilde = np.linalg.inv(K_indu_indu_inv + dt / (2 * z * kT) * K_indu_indu_inv @ K_data_indu.T @ K_data_indu @ K_indu_indu_inv)
            K_tilde_chol = np.linalg.cholesky(K_tilde + eps * np.eye(num_indu ** num_dims))
        else:
            # if v == 0, then K matrices are the the same each iteration
            K_data_indu = variables.K_data_indu
            K_tilde = variables.K_tilde
            K_tilde_chol = variables.K_tilde_chol

        # Calculate mean
        mu_tilde = 1 / (2 * kT) * (K_tilde @ (K_indu_indu_inv @ (K_data_indu.T @ y_data)))

        # Sample U
        u_indu = (
            mu_tilde + K_tilde_chol @ np.random.randn(num_indu ** num_dims, 1)
        )

        # calculate force and U_grid
        u_grid = (K_grid_indu @ (K_indu_indu_inv @ u_indu)).reshape([num_grid] * num_dims, order='F')
        f_data = np.zeros((num_data, num_dims))
        f_data[ids, :] = (K_data_indu @ (K_indu_indu_inv @ u_indu)).reshape((-1, num_dims), order='F')

        # load variables
        variables.u_indu = u_indu
        variables.u_grid = u_grid
        variables.f_data = f_data

        return

    @staticmethod
    def sample_trajectory(data, variables):

        # Done sample trajectory if there is no noise
        if variables.v == 0:
            return

        if np.random.rand() < .8:
            PotentialLearner.sample_trajectory_MH(data, variables)
        else:
            PotentialLearner.sample_trajectory_HMC(data, variables)

        return

    @staticmethod
    def sample_trajectory_HMC(data, variables, num_per_sec=50):

        # get variables
        z = variables.z
        v = variables.v
        dt = variables.dt
        kT = variables.kT
        x0_mean = variables.x0_mean
        x0_vars = variables.x0_vars
        x_data = variables.x_data
        x_indu = variables.x_indu
        u_indu = variables.u_indu
        f_data = variables.f_data
        kernel = variables.kernel
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_traj = variables.num_traj
        K_indu_indu_inv = variables.K_indu_indu_inv
        traj_mask = variables.traj_mask
        parallelize = variables.parallelize

        # set up variables
        x_data = x_data.copy()
        f_data = f_data.copy()
        kick = 2 * dt * kT / z
        K_inv_U = (K_indu_indu_inv @ u_indu).reshape(-1, 1, order='F')

        def sample_positions(ids, x_data, f_data):

            # Select variables
            n0 = ids[0]
            nf = ids[-1]
            t = traj_mask[n0]
            num_ids = len(ids)
            data_ids = data[ids, :]

            # Set up x0 and kicks
            x_prev = np.zeros((num_ids, num_dims))
            kicks = kick * np.ones((num_ids, num_dims))
            if (n0 == 0) or traj_mask[n0-1] != t:
                x_prev[0, :] = x0_mean
                kicks[0, :] = x0_vars
            else:
                x_prev[0, :] = x_data[n0 - 1, :] + dt / z * f_data[n0 - 1, :]

            # Set up x final
            if (nf == num_data-1) or traj_mask[nf+1] != t:
                C = np.zeros((num_ids, num_ids))
                x_next = np.zeros((num_ids, num_dims))
            else:
                C = np.diag([0] * (num_ids - 1) + [1])
                x_next = np.zeros((num_ids, num_dims))
                x_next[-1, :] = x_data[nf + 1, :]

            # set up dynamics matrixes
            B = np.diag(np.ones(num_ids - 1), -1)
            A = np.eye(num_ids) - B

            # Reshape
            data_ids = data_ids.reshape((-1, 1), order='F')
            kicks = kicks.reshape((-1, 1), order='F')
            x_prev = x_prev.reshape((-1, 1), order='F')
            x_next = x_next.reshape((-1, 1), order='F')
            C = np.kron(np.eye(num_dims), C)
            B = np.kron(np.eye(num_dims), B)
            A = np.kron(np.eye(num_dims), A)

            # HMC proposal distribution
            h = stats.expon.rvs(scale=.025)
            M = np.ones((num_ids * num_dims, 1))
            M_inv = 1 / M
            num_steps = stats.poisson.rvs(mu=25)

            def dp_dh(q_):
                x_ = q_.reshape((-1, num_dims), order='F')
                f = - kT * kernel(x_, x_indu, d_dx=True) @ K_inv_U
                df = np.diag((- kT * kernel(x_, x_indu, dd_ddx=True) @ K_inv_U)[:, 0])
                B_f = np.vstack([np.zeros((1, 1)), f[:-1, :]])  # faster calculation of B @ f
                B_df = np.diag(df[:-1, 0], -1)  # faster calculation of B @ df
                y_ = (
                    - (q_ - data_ids) / v
                    - (A - dt / z * B_df).T @ (A @ q_ - dt / z * B_f - x_prev) / kick
                    - (C + dt / z * C @ df).T @ (C @ (q_ + dt / z * f) - x_next) / kick
                )
                return y_

            def probability(q_, p_):
                x_ = q_.reshape((-1, num_dims), order='F')
                f = - kT * kernel(x_, x_indu, d_dx=True) @ K_inv_U
                prob = (
                    np.sum(stats.norm.logpdf(data_ids, loc=q_, scale=np.sqrt(v)))                         # likelihood
                    + np.sum(stats.norm.logpdf(q_, loc=B @ (q_ + dt / z * f) + x_prev, scale=np.sqrt(kick)))  # prior on x
                    + np.sum(stats.norm.logpdf(x_next, loc=C @ (q_ + dt / z * f), scale=np.sqrt(kick)))       # prior on final x
                    + np.sum(stats.norm.logpdf(p_, loc=0, scale=np.sqrt(M)))                              # prior on p
                )
                return prob

            # Run HMC
            q = x_data[ids, :].copy().reshape((-1, 1), order='F')
            p = stats.norm.rvs(loc=np.zeros(q.shape), scale=np.sqrt(M))
            P_old = probability(q, p)
            for _ in range(num_steps):
                p = p + h / 2 * dp_dh(q)
                q = q + h * p * M_inv
                p = p + h / 2 * dp_dh(q)
            P_new = probability(q, p)

            # accept or reject
            acc_prob = P_new - P_old
            if acc_prob > np.log(np.random.rand()):
                x_data_ids = q[:, :].reshape((-1, num_dims), order='F')
                f_data_ids = (- kT * kernel(x_data_ids, x_indu, d_dx=True) @ K_inv_U).reshape((-1, num_dims), order='F')
            else:
                x_data_ids = x_data[ids, :].copy()
                f_data_ids = f_data[ids, :].copy()

            return ids, x_data_ids, f_data_ids

        # Sample trajectories
        acceptance_rate = np.zeros(2)
        for t in range(num_traj):

            # Split trajectory into sections
            idt = np.where(traj_mask == t)[0]
            section_ids = np.zeros(len(idt), dtype=int)
            section_ids[range(0, len(idt), num_per_sec)] += 1
            section_ids[:] = np.cumsum(section_ids) - 1
            num_sections = section_ids[-1] + 1

            # Loop through sampling two sections at a time while leaving the third constant
            for i in range(3):
                sections = [np.where((section_ids == j) | (section_ids == j + 1))[0] for j in range(i, num_sections, 3)]
                sections = [idt[ids] for ids in sections]
                
                # Sample each section
                if parallelize:
                    results = Parallel(n_jobs=-1)(
                        delayed(sample_positions)(ids, x_data, f_data) for ids in sections
                    )
                else:
                    results = [sample_positions(ids, x_data, f_data) for ids in sections]

                # Update positions
                for ids, x_new, f_new in results:
                    acceptance_rate += x_new.size, np.sum(x_new == x_data[ids, :])
                    x_data[ids, :] = x_new
                    f_data[ids, :] = f_new

        # update variables
        variables.x_data = x_data
        variables.f_data = f_data

        # print acceptance ratio
        print('(Hx{}%)'.format(round(100 * acceptance_rate[1]/acceptance_rate[0])), end='')

        return

    @staticmethod
    def sample_trajectory_MH(data, variables):

        z = variables.z
        v = variables.v
        dt = variables.dt
        kT = variables.kT
        x0_mean = variables.x0_mean
        x0_vars = variables.x0_vars
        x_data = variables.x_data
        x_indu = variables.x_indu
        x_prop_std = variables.x_prop_std
        f_data = variables.f_data
        u_indu = variables.u_indu
        kernel = variables.kernel
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_traj = variables.num_traj
        K_indu_indu_inv = variables.K_indu_indu_inv
        traj_mask = variables.traj_mask
        parallelize = variables.parallelize

        x_data = x_data.copy()
        f_data = f_data.copy()
        kick = 2 * dt * kT / z
        K_inv_U = K_indu_indu_inv @ u_indu.reshape(-1, 1)

        def sample_position(n, x_data, f_data):

            # Select variables
            t = traj_mask[n]
            if (n == 0) or traj_mask[n-1] != t:
                mu_prev = x0_mean
                sigma_prev = np.sqrt(x0_vars)
            else:
                mu_prev = x_data[n-1, :] + dt/z*f_data[n-1, :]
                sigma_prev = np.sqrt(kick)
            if (n == num_data-1) or traj_mask[n+1] != t:
                x_next = None
            else:
                x_next = x_data[n+1, :]

            # Create probability function
            def probability(x_, f_):
                prob = (
                    np.sum(stats.norm.logpdf(data[n, :], loc=x_, scale=np.sqrt(v)))
                    + np.sum(stats.norm.logpdf(x_, loc=mu_prev, scale=sigma_prev))
                )
                if x_next is not None:
                    prob += np.sum(stats.norm.logpdf(x_next, loc=x_+dt/z*f_, scale=np.sqrt(kick)))
                return prob

            # Propose new position
            x_old = x_data[n, :].copy()
            f_old = f_data[n, :].copy()
            x_new = x_old + x_prop_std * np.random.randn(num_dims)
            f_new = (- kT * kernel(x_new[None, :], x_indu, d_dx=True) @ K_inv_U)[:, 0]
            # calculate acceptance probability
            P_old = probability(x_old, f_old)
            P_new = probability(x_new, f_new)
            acc_prob = P_new - P_old
            if np.isnan(acc_prob):
                print('ohno')
            # sample new or old
            if acc_prob > np.log(np.random.rand()):
                x_out = f_new
                f_out = x_new
            else:
                x_out = x_old
                f_out = f_old

            return n, x_out, f_out

        # Paralellize sampling trajectories
        ratio = np.zeros(2)  # total, accepted
        for m in range(2):
            if parallelize:
                results = Parallel(n_jobs=-1)(
                    delayed(sample_position)(n, x_data, f_data) for n in range(m, num_data, 2)
                )
            else:
                results = [sample_position(n, x_data, f_data) for n in range(m, num_data, 2)]
            for n, x_new, f_new in results:
                ratio += 1, np.all(x_new == x_data[n, :])
                x_data[n, :] = x_new
                f_data[n, :] = f_new

        # update variables
        variables.x_data = x_data
        variables.f_data = f_data

        # print acceptance ratio
        print('(Mx{}%)'.format(round(100 * ratio[1]/ratio[0])), end='')

        return

    @staticmethod
    def sample_noise(data, variables):

        if variables.v == 0:
            return

        x_data = variables.x_data
        v_shape = variables.v_shape
        v_scale = variables.v_scale
        num_data = variables.num_data
        num_dims = variables.num_dims

        shape = v_shape + num_dims * num_data / 2
        scale = v_scale + np.sum((data - x_data) ** 2) / 2

        v = stats.invgamma.rvs(a=shape, scale=scale)
        variables.v = v

        return

    @staticmethod
    def sample_friction(data, variables):

        dt = variables.dt
        kT = variables.kT
        x_data = variables.x_data
        f_data = variables.f_data
        traj_mask = variables.traj_mask
        num_traj = variables.num_traj
        z_shape = variables.z_shape
        z_scale = variables.z_scale
        x_prop_shape = variables.x_prop_shape
        x0_mean = variables.x0_mean
        x0_vars = variables.x0_vars

        # Define probability function
        def probability(z_):
            prob = 0
            for t in range(num_traj):
                ids = np.where(traj_mask == t)[0]
                x0 = x_data[ids[0], :]
                dx = x_data[ids[1:], :] - x_data[ids[:-1], :]
                f = f_data[ids[:-1], :]
                kick = 2 * dt * kT / z_
                prob += (
                    stats.gamma.logpdf(z_, a=z_shape, scale=z_scale)
                    + np.sum(stats.norm.logpdf(dx, loc=dt*f/z_, scale=np.sqrt(kick)))
                    + np.sum(stats.norm.logpdf(x0, loc=x0_mean, scale=np.sqrt(x0_vars)))
                )
            return prob

        # Propose and sample friction multiple times
        z = variables.z
        ratio = np.zeros(2)  # total, accepted
        for _ in range(10):
            z_old = z
            z_new = stats.gamma.rvs(a=x_prop_shape, scale=z_old/x_prop_shape)
            P_old = probability(z_old)
            P_new = probability(z_new)
            acc_prob = (
                P_new - P_old
                + stats.gamma.logpdf(z_old, a=x_prop_shape, scale=z_new/x_prop_shape)
                - stats.gamma.logpdf(z_new, a=x_prop_shape, scale=z_old/x_prop_shape)
            )
            if acc_prob > np.log(np.random.rand()):
                z = z_new
            ratio += 1, np.all(z == z_old)

        # Print acceptance ratio
        print('(z{}%)'.format(round(100 * ratio[1]/ratio[0])), end='')

        # Update variables
        variables.z = z

        return

    @staticmethod
    def posterior(data, variables, **kwargs):

        # set kwarg args into variables
        variables = copy.deepcopy(variables)
        for key, val in kwargs:
            setattr(variables, key, val)

        # extract variables
        z = variables.z
        v = variables.v
        dt = variables.dt
        kT = variables.kT
        x0_mean = variables.x0_mean
        x0_vars = variables.x0_vars
        v_shape = variables.v_shape
        v_scale = variables.v_scale
        z_shape = variables.z_shape
        z_scale = variables.z_scale
        u_indu = variables.u_indu
        f_data = variables.f_data
        x_data = variables.x_data
        K_indu_indu = variables.K_indu_indu
        num_traj = variables.num_traj
        num_indu = variables.num_indu
        num_dims = variables.num_dims
        traj_mask = variables.traj_mask

        # calculate constants
        kick = 2 * dt * kT / z

        # calculate posterior
        prob = (
            stats.gamma.logpdf(z, a=z_shape, scale=z_scale)              # prior on z
            + stats.multivariate_normal.logpdf(
                u_indu[:, 0], 
                np.zeros(num_indu ** num_dims), 
                cov=K_indu_indu+.1*np.eye(num_indu ** num_dims)
            )
        )
        for t in range(num_traj):
            ids = np.where(traj_mask == t)[0]
            x0 = x_data[ids[0], :]
            dx = x_data[ids[1:], :] - x_data[ids[:-1], :]
            f = f_data[ids[:-1], :]
            prob += (
                np.sum(stats.norm.logpdf(x0, loc=x0_mean, scale=np.sqrt(x0_vars)))  # prior on x[0, :]
                + np.sum(stats.norm.logpdf(dx, loc=dt*f/z, scale=np.sqrt(kick)))    # prior on x[1:, :]
            )
        if v > 0:
            prob += (
                np.sum(stats.norm.logpdf(data, loc=x_data, scale=np.sqrt(v)))  # likelihood
                + stats.invgamma.logpdf(v, a=v_shape, scale=v_scale)           # prior on v
            )

        return prob

    @staticmethod
    def plot_variables(data, variables, groundtruth=None):

        if groundtruth is not None:
            gt = SimpleNamespace(**{**PARAMETERS, **groundtruth})

        dt = variables.dt
        kT = variables.kT
        x_data = variables.x_data
        x_grid = variables.x_grid
        u_indu = variables.u_indu
        K_grid_indu = variables.K_grid_indu
        K_indu_indu_inv = variables.K_indu_indu_inv
        num_data = variables.num_data
        num_dims = variables.num_dims
        num_grid = variables.num_grid
        num_traj = variables.num_traj
        traj_mask = variables.traj_mask

        times = np.arange(num_data) * dt
        u_grid = (K_grid_indu @ K_indu_indu_inv @ u_indu).reshape(-1, order='F')
        u_grid -= np.min(u_grid)

        if num_dims == 1:

            fig = plt.gcf()
            fig.clf()
            fig.set_size_inches(12, 6)
            plots = np.empty((1, 2), dtype=object)
            gs = gridspec.GridSpec(nrows=1, ncols=4, figure=fig)
            plots[0, 0] = fig.add_subplot(gs[0, :-1])
            plots[0, 1] = fig.add_subplot(gs[0, -1], sharey=plots[0, 0])

            xplot = plots[0, 0]
            xplot.set_title('Trajectory')
            xplot.set_xlabel('time (ns)')
            xplot.set_ylabel('position (nm)')
            for t in range(num_traj):
                idt = np.where(traj_mask == t)[0]
                xplot.plot(dt * idt, data[idt, :], color='g', label='Data')
                if num_traj > 1:
                    xplot.axvline(dt * idt[0], color='k', linewidth=2, label='New trajectory')
                if groundtruth is not None:
                    if gt.x_data is not None:
                        xplot.plot(dt * idt, gt.x_data[idt, :], color='r', label='Ground truth')
                xplot.plot(dt * idt, x_data[idt, :], color='b', label='Sampled trajectory')
            handles, labels = xplot.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            xplot.legend(by_label.values(), by_label.keys())

            Uplot = plots[0, 1]
            Uplot.set_title('Potential')
            Uplot.set_xlabel('potential (kT)')
            if groundtruth is not None:
                if gt.force is not None:
                    U_gt = - np.cumsum(gt.force(x_grid)) * (x_grid[1] - x_grid[0]) / kT
                    U_gt -= np.min(U_gt)
                    Uplot.plot(U_gt, x_grid, color='r', label='Ground truth')
            Uplot.plot(u_grid, x_grid, color='b', label='Sampled potential')
            Uplot.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            if np.max(u_grid) - np.min(u_grid) > 0:
                Uplot.set_xlim([np.min(u_grid), np.max(u_grid)])

        elif num_dims == 2:

            fig = plt.gcf()
            plt.clf()
            ax = fig.add_subplot(111)
            # ax = fig.add_subplot(111, projection='3d')

            X = x_grid[:num_grid, 1]
            X, Y = np.meshgrid(X, X)
            Z = (K_grid_indu @ K_indu_indu_inv @ u_indu).reshape(X.shape, order='F')

            ax.imshow(Z)

            # ax.plot_surface(X, Y, Z, cmap='viridis', alpha=.5)
            # ax.set_title('Potential')
            # ax.set_xlabel('position (nm)')
            # ax.set_ylabel('position (nm)')
            # ax.set_zlabel('potential (kT)')

        plt.tight_layout()
        plt.pause(.1)

        return

    @staticmethod
    def plot_results(data, history, groundtruth=None):

        # set up gt
        if groundtruth is not None:
            gt = SimpleNamespace(**{**PARAMETERS, **groundtruth})

        # set up MAP
        MAP = history.get('MAP')
        probs = history.get('P')
        last = min([*np.where(probs == 0)[0], probs.shape[0]])
        burn = int(last / 2)

        # get variables
        dt = MAP.dt
        kT = MAP.kT
        x_data = MAP.x_data
        x_grid = MAP.x_grid
        u_indu = MAP.u_indu
        K_grid_indu = MAP.K_grid_indu
        K_indu_indu_inv = MAP.K_indu_indu_inv
        num_data = MAP.num_data
        num_dims = MAP.num_dims
        num_traj = MAP.num_traj
        traj_mask = MAP.traj_mask

        if num_dims == 2:
            print('cant deal with 2D yet') # todo: make 2d work
            return

        # calculate values
        times = np.arange(num_data) * dt
        # calculate potential
        u_indu_history = history.get('u_indu', burn=burn, last=last)
        u_indu_mean = np.mean(u_indu_history, axis=0)
        u_indu_std = np.std(u_indu_history, axis=0)
        u_grid_mean = K_grid_indu @ K_indu_indu_inv @ u_indu_mean
        u_grid_std = K_grid_indu @ K_indu_indu_inv @ u_indu_std
        # calculate trajectory
        x_data_history = history.get('x_data', burn=burn, last=last)
        x_data_mean = np.mean(x_data_history, axis=0)
        x_data_std = np.std(x_data_history, axis=0)


        fig = plt.gcf()
        fig.clf()
        fig.set_size_inches(12, 6)
        plots = np.empty((1, 2), dtype=object)
        gs = gridspec.GridSpec(nrows=1, ncols=4, figure=fig)
        plots[0, 0] = fig.add_subplot(gs[0, :-1])
        plots[0, 1] = fig.add_subplot(gs[0, -1], sharey=plots[0, 0])

        xplot = plots[0, 0]
        xplot.set_title('Trajectory')
        xplot.set_xlabel('time (ns)')
        xplot.set_ylabel('position (nm)')
        for t in range(num_traj):
            idt = np.where(traj_mask == t)[0]
            xplot.axvline(dt * idt[0], color='k', linewidth=2, label='New trajectory')
            xplot.plot(dt * idt, x_data_mean[idt], color='b', label='Inferred trajectory')
            xplot.fill_between(
                dt * idt,
                (x_data_mean - x_data_std)[idt],
                (x_data_mean + x_data_std)[idt],
                color='skyblue',
                alpha=.5,
                label='Uncertainty',
            )
            if groundtruth is not None:
                if gt.x_data is not None:
                    xplot.plot(dt * idt, gt.x_data[idt, :], color='r', linewidth=.5, label='Ground truth')
        handles, labels = xplot.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        xplot.legend(by_label.values(), by_label.keys())

        uplot = plots[0, 1]
        uplot.set_title('Potential')
        uplot.set_xlabel('potential (kT)')
        uplot.plot(u_grid_mean, x_grid, color='b', label='Inferred potential')
        uplot.fill_betweenx(
            x_grid[:, 0],
            u_grid_mean - u_grid_std,
            u_grid_mean + u_grid_std,
            color='skyblue',
            alpha=.5,
            label='Uncertainty',
        )
        if groundtruth is not None:
            if gt.force is not None and num_dims == 1:
                U_gt = np.cumsum(gt.force(x_grid[::-1]))[::-1] * (x_grid[1] - x_grid[0]) / kT
                U_gt += u_grid_mean[np.argmin(np.abs(x_grid))] - U_gt[np.argmin(np.abs(x_grid))]
                uplot.plot(U_gt, x_grid, color='r', label='Ground truth')

        uplot.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        # if np.max(u_grid_mean) - np.min(u_grid_mean) > 0:
        #     uplot.set_xlim([np.min(u_grid_mean), np.max(u_grid_mean)])

        plt.tight_layout()
        plt.pause(.1)

        return

    @staticmethod
    def plot_x_results(data, history, tlims=[0, None], burn=.5, ax=None, groundtruth=None, inset=True, legend=False):

        # set up ground truth
        if groundtruth is not None:
            gt = SimpleNamespace(**{**PARAMETERS, **groundtruth})

        MAP = history.get('map')
        probs = history.get('P')
        last = min([*np.where(probs == 0)[0], probs.shape[0]])
        burn = int(last * burn)
        dt = MAP.dt
        num_data = MAP.num_data

        if 0 < burn < 1:
            if last is None:
                burn = round(burn * history.num_iter)
            else:
                burn = round(burn * last)

        if ax is None:
            fig, plots = plt.subplots(1, 1)
            fig.set_size_inches(9, 6)
        else:
            plots = np.empty((1, 1), dtype=object)
            plots[0, 0] = ax


        # calculate x
        times = dt * np.arange(num_data) / 1e6  # 1e6 ns = 1 ms
        x_mean = MAP.x_data
        # x_mean = np.mean(history.get('x_data', burn=burn, last=last), axis=0)
        # x_std = np.std(history.get('x_data', burn=burn, last=last), axis=0)

        # rescale
        # data = data[tlims[0]:tlims[1]]
        # times = times[tlims[0]:tlims[1]]
        # x_mean = x_mean[tlims[0]:tlims[1]]
        # x_std = x_std[tlims[0]:tlims[1]]

        # plot x
        xplot = plots[0, 0]
        xplot.plot(data, times, color='g', label='Data')
        if groundtruth is not None:
            if gt.x_data is not None:
                xplot.plot(gt.x_data, times, color='r', label='Ground truth')
        # xplot.fill_betweenx(times, x_mean - x_std, x_mean + x_std, color='b', alpha=.1, label='Inferred uncertainty')
        xplot.plot(x_mean, times, color='b', label='Inferred trajectory')

        # zoom into x
        #tlims = [10,20]
        data = data[tlims[0]:tlims[1]]
        times = times[tlims[0]:tlims[1]]
        x_mean = x_mean[tlims[0]:tlims[1]]
        # x_std = x_std[tlims[0]:tlims[1]]

        if inset:
            axins = inset_axes(xplot, .5, .5, loc=1,)# transform=xplot.figure.transFigure)
            mark_inset(xplot, axins, loc1=3, loc2=4, fc="none", ec="0.5")
            # axins.set_ylim(tlims)
            axins.plot(data, times, color='g', label='Data')
            axins.plot(data[tlims[0]:tlims[1]], times[tlims[0]:tlims[1]], color='g', label='Data')
            if gt.x_data is not None:
                axins.plot(gt.x_data[tlims[0]:tlims[1]], times, color='r', label='Ground truth')
            # axins.fill_betweenx(times, x_mean - x_std, x_mean + x_std, color='b', alpha=.5, label='Inferred uncertainty')
            axins.plot(x_mean, times, color='b', label='Inferred trajectory')
            axins.set_xticks([])
            axins.set_yticks([])

        if legend:
            xplot.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        if ax is None:
            xplot.set_title('Trajectory')
            xplot.set_ylabel('time (ms)')
            xplot.set_xlabel('position (nm)')
            plt.tight_layout()
            plt.pause(.1)

        return

    @staticmethod
    def plot_U_results(data, history, burn=.5, ax=None, compare=True, nonoise=None, groundtruth=None, legend=False):

        # set up ground truth
        if groundtruth is not None:
            gt = SimpleNamespace(**{**PARAMETERS, **groundtruth})

        # set up variables
        MAP = history.get('map')
        probs = history.get('P')
        last = min([*np.where(probs == 0)[0], probs.shape[0]])
        burn = int(last * burn)
        kT = MAP.kT
        x_grid = MAP.x_grid
        x_indu = MAP.x_indu
        K_grid_indu = MAP.K_grid_indu
        K_indu_indu_inv = MAP.K_indu_indu_inv
        num_grid = MAP.num_grid
        num_dims = MAP.num_dims

        if ax is None:
            fig = plt.gcf()
            fig.clf()
            fig.set_size_inches(9, 6)
            plots = np.empty((1, 1), dtype=object)
            gs = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)
            if num_dims == 1:
                plots[0, 0] = fig.add_subplot(gs[0, 0])
            elif num_dims == 2:
                plots[0, 0] = fig.add_subplot(gs[0, 0], projection='3d')
        else:
            plots = np.empty((1, 1), dtype=object)
            plots[0, 0] = ax

        # plot U
        Uplot = plots[0, 0]
        if num_dims == 1:

            # Find zero id
            i0 = np.argmin(np.abs(x_grid))

            # Calculate U from history
            u_hist_indu = history.get('u_indu', burn=burn, last=last)
            u_hist = K_grid_indu @ K_indu_indu_inv @ u_hist_indu[:, :, 0].T
            u_hist -= u_hist[i0, :]
            u_mean = np.mean(u_hist, axis=1)
            u_std = np.std(u_hist, axis=1)


            # calculate U with residence time
            u_comp, x_comp = np.histogram(data, density=True)
            u_comp = - np.log(u_comp)  # not time kT because units are kT
            u_comp += u_mean[np.argmin(np.abs(x_grid))] - u_comp[np.argmin(np.abs(x_comp))]
            x_comp = x_comp[:-1] + .5 * (x_comp[1] - x_comp[1])

            if groundtruth is not None:
                u_gt = gt.potential(x_grid)
                u_gt -= u_gt[i0]
                Uplot.plot(x_grid[:, 0], u_gt, color='r', label='Ground truth')
            if compare:
                Uplot.step(x_comp, u_comp, where='posteriors', color='g', label='Boltzmann potential')
            Uplot.fill_between(x_grid[:, 0], u_mean - u_std, u_mean + u_std, color='b', alpha=.1, label='Inferred uncertainty')
            Uplot.plot(x_grid[:, 0], u_mean, color='b', label='Inferred potential')
            if ax is None:
                Uplot.set_title('Potential')
                Uplot.set_xlabel('potential (kT)')
                Uplot.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                plt.tight_layout()
                plt.pause(.1)
            if legend:
                Uplot.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        return

    @staticmethod
    def learn_potential(data, parameters=None, num_iter=1000,
                        saveas=None, plot_status=False, groundtruth=None, log=False, **kwargs):
        
        # Print status
        print("Starting inference")
        
        # Set up log
        if not log:
            pass
        elif log is True:
            if saveas is not None:
                log = saveas.split('/')[-1] + '.log'
            else:
                log = f'log{np.random.randint(1e6)}.log'
        elif not log.lower().endswith('.log'):
            log = f'{log}.log'

        # Reformat data
        data = np.atleast_2d(data)  # data is 2D (time index, dimension index)
        if data.shape[1] > data.shape[0]:
            data = data.T

        # Set up parameters
        if parameters is None:
            parameters = {**PARAMETERS, **kwargs}
        else:
            parameters = {**PARAMETERS, **parameters, **kwargs}

        # Set up variables
        variables = PotentialLearner.initialize_variables(data, parameters)
        print('Parameters:')
        for key in sorted(parameters.keys()):
            text = str(getattr(variables, key)).replace('\n', ', ')
            text = '--{} = {}'.format(key, text)
            if len(text) > 80: text = text[:77] + '...'
            print(text)
            if log:
                with open(log, 'a') as handle:
                    handle.write(text + '\n')
                
        # Set up history
        MAP = copy.deepcopy(variables)
        if saveas is not None:
            history = HistoryH5(
                save_name=saveas,
                variables=variables,
                num_iter=num_iter,
                fields=[
                    'P',
                    'v', 
                    'z', 
                    'u_indu',
                    # 'x_data',
                ],
            )

        # Run the Gibbs sampler
        for i in range(num_iter):

            # Print status
            print(f'Iteration {i+1} of {num_iter} [', end='')
            t = time.time()

            # Sample variables
            PotentialLearner.sample_potential(data, variables)
            print('%', end='')
            PotentialLearner.sample_trajectory(data, variables)
            print('%', end='')
            PotentialLearner.sample_noise(data, variables)
            print('%', end='')
            PotentialLearner.sample_friction(data, variables)
            print('%', end='')

            # Plot
            if plot_status:
                PotentialLearner.plot_variables(data, variables, groundtruth)
                print('%', end='')

            # Save sample
            variables.P = PotentialLearner.posterior(data, variables)
            if variables.P >= MAP.P:
                MAP = copy.deepcopy(variables)
            if saveas is not None:
                history.checkpoint(variables, i)

            # Print status
            print(f'%] ({(time.time()-t):.2f} s) (prob={variables.P:.3e})'.format())
            if log:
                with open(log, 'a') as handle:
                    handle.write(f'Iteration {i+1} of {num_iter} ({round(time.time()-t, 2)}s)\n')

        # Return output
        print('Sampling complete')
        if saveas is not None:
            return MAP, history
        else:
            return MAP


