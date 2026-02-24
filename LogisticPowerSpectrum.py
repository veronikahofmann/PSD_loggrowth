import matplotlib.pyplot as plt
import numpy as np
import time
import math


def log_power_spectrum(params, wavelengths, extra_params):
    """
    Computes the power spectral density (PSD) of the RD-equation with logistic growth. The PSD is normed with P_0.
    :param params: array of input parameters: [gamma, D, K]
    :param wavelengths: array of frequencies k for which the PSD P(k) should be calculated.
    :param extra_params: array of additional parameters: [P_0, 2*k_max, N], where P_0 is the normalization constant,
    k_max is the maximum frequency up to which the PSD should be calculated, and N is the number of spatial grid points.
    :return: the PSD P(k) as an array evaluated at the input-wavelengths k.
    """
    gamma = params[0]
    D = params[1]
    K = 1  # K = params[2]

    P_0 = extra_params[0]

    # discretization parameters
    N = extra_params[2]  # no. of grid points
    dt = 0.01  # time step size
    t_max = 1  # final time
    tol = 10e-6  # convergence tolerance

    # Fourier space grid
    k_max = extra_params[1]  # define the maximum value for k (for positive values of k only, use half of the maximum k you're interested in)
    k = np.linspace(0, 2*k_max, N)  # discretized k-space (for positive values of k only, use 2*k_max as limit)

    # initialize the solution (the hat symbolizes the (1D) Fourier transform)
    u_hat_n = np.ones(N)  # i. c.  u_hat(k,0) = exp((...)*0) * 1
    u_hat_alln = np.zeros((int(np.ceil(t_max/dt)) + 1, N))
    all_convs = np.zeros((int(np.ceil(t_max/dt)) + 1, N))

    # loop in time
    current_time_iter = 0
    for t in np.arange(0, t_max + dt, dt):
        # iteration variable for convergence
        u_hat_n1 = u_hat_n
        converged = False

        # fixed-point iteration for each time step
        while not converged:
            # compute the linear part of the update in the case u_hat(k,0) = 1
            u_hat_lin = np.exp((gamma - D * k ** 2) * t)  # 1xN vector

            # convolution computation
            if t == 0:
                conv_term = 2 * k_max * np.ones(N)
            else:
                conv_term = np.convolve(u_hat_n, u_hat_n, 'same') * (2 * k_max / (N-1))

            # time integration for the convolution term (composite Simpson)
            integral = np.zeros(N)
            for i in range(N):
                if t == 0:
                    # for t = 0 the integral should be zero since there is no area under the curve
                    integral[i] = 0
                elif t == dt:
                    # for t = dt, use the trapezoidal rule since there is only one interval
                    integral[i] = (dt / 2) * (all_convs[1, i] + np.exp(-(gamma - D * k[i] ** 2) * t) * conv_term[i])
                else:
                    num_intervals = t / dt - 1  # calculate the number of intervals, excluding the last point

                    # apply Simpson's rule
                    simpson_integral = 0
                    for tau in np.arange(dt, (num_intervals * dt) + dt, dt):
                        t_index = int(tau / dt)
                        f_tau = np.exp(-(gamma - D * k[i] ** 2) * tau) * all_convs[t_index, i]
                        if t_index % 2 == 0:  # even index
                            simpson_integral = simpson_integral + 2 * f_tau
                        else:  # odd index
                            simpson_integral = simpson_integral + 4 * f_tau

                    # final Simpson's rule computation
                    integral[i] = (dt / 3) * (
                                all_convs[1, i] + simpson_integral + np.exp(-(gamma - D * k[i] ** 2) * t) * conv_term[i])

            # Crank-Nicholson update
            u_hat_lin = (u_hat_lin - 0.5 * (gamma / K) * np.exp((gamma - D * k ** 2) * t) * integral + 0.5 * u_hat_n1) / (1 + 0.5 * (gamma / K) * np.exp((gamma - D * k ** 2) * t) * integral)

            # check for convergence
            l_infty = max(abs(u_hat_lin - u_hat_n1))
            if l_infty < tol:
                converged = True
            elif math.isnan(l_infty):
                raise TypeError('NaN observed for gamma = ' + str(gamma) + ', D = ' + str(D) + ', K = ' + str(K) +
                                ' => no convergence')

            # update for the next iteration
            u_hat_n1 = u_hat_lin

        # save result of the current time step
        u_hat_alln[current_time_iter, :] = u_hat_n1
        all_convs[current_time_iter, :] = conv_term

        # update for the next time step
        u_hat_n = u_hat_n1
        current_time_iter += 1

    # match the values in wavelengths to the values of k
    k_min_dist = np.zeros_like(wavelengths)  # array of k-values with the lowest distance to the entries of wavelengths
    k_min_dist_ind = np.zeros_like(wavelengths, dtype=int)  # indices
    for i in range(len(wavelengths)):
        idx = np.argmin(np.abs(k - wavelengths[i]))  # finds the index idx of the element in k that is closest to wavelengths(i)
        k_min_dist[i] = k[idx]
        k_min_dist_ind[i] = int(idx)

    u_hat_wavelengths = u_hat_n1[k_min_dist_ind]
    P_raw = np.abs(u_hat_wavelengths) ** 2
    P = P_0 * P_raw/max(P_raw)  # TODO: lieber P_raw[0] verwenden statt max?

    return P


# check the function
if __name__ == "__main__":
    params = [0.3, 0.05, 1]  # [gamma, D, K]  # [1.1, 0.12, 1]   # [6, 20, 1]  # [2.9609, 1.0704, 1]  # [3, 1, 1]  # [7.0, 100000.0, 1]  #
    extra_params = [1, 3, 1000]  # [P_0, 2*k_max, N]  # [700000, 3, 1000]  # [9.98432870059305, 3, 1000]  # 9.98432870059305  #
    wavelengths = np.logspace(-3, 1, 1000)

    start = time.time()
    psd = log_power_spectrum(params, wavelengths, extra_params)
    end = time.time()

    print(f'Execution took {end - start} seconds.')

    plt.loglog(wavelengths, psd)
    plt.grid(visible=True)
    plt.xlabel('frequency k')
    plt.ylabel('PSD')
    # plt.ylim((6.5, 10))
    plt.show()
