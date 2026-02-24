% LogisticPowerSpectrum.m

% parameters
wavelengths = logspace(-3, 1, 1000);
gamma = 5.35;
D = 0.2;
K = 1;
t_max = 1;

% computations and plot
tic
ps = power_spectrum(D, gamma, K, wavelengths, t_max);
toc

figure;
loglog(wavelengths, ps);
grid on
xlabel('frequency $k$', 'Interpreter', 'latex');
ylabel('$\hat{P}$', 'Interpreter', 'latex');


function P = power_spectrum(D, gamma, K, wavelengths, t_max)
    P_0 = 1;

    % values of the discretization parameters
    N = 1000; %1000;  %500 % no. grid points
    dt = 0.01;  % time step size

    if nargin < 5
        t_max = 1;  % final time
    end
    tol = 10e-6;  % convergence tolerance
    
    % Fourier space grid
    k_max = 3; %5;    % Define the maximum value for k (for positive values of k only, use half of the maximum k you're interested in)
    k = linspace(0, 2*k_max, N);  %k = linspace(-k_max, k_max, N);  % Discretized k-space (for positive values of k only, use 2*k_max as limit)
    
    % initialize solution (the hat stands for (1D) Fourier transform)
    u_hat_n = ones(1, N);  % i. c.  u_hat(k,0) = exp((...)*0) * 1
    u_hat_alln = zeros(ceil(t_max/dt) + 1, N);
    all_convs = zeros(ceil(t_max/dt) + 1, N);
    
    % loop in time
    current_time_it = 0;
    for t = 0:dt:t_max
        % iteration variable for convergence
        u_hat_n1 = u_hat_n;
        converged = false;
        
        % fixed-point iteration for each time step
        while ~converged
            
            % compute the linear part of the update in the case u_hat(k,0) = 1
            u_hat_lin = exp((gamma - D * k.^2) * t);  % 1xN vector
    
            % convolution computation
            if t == 0
                conv_term = 2*k_max*ones(1, N);
            else
                conv_term = conv(u_hat_n, u_hat_n, 'same') * (2 * k_max / (N-1));
            end
            
            % time integration for the convolution term (composite Simpson)
            integral = zeros(1, N);
            for i = 1:N
                if t == 0
                    % For t = 0, the integral should be zero since there's no area under the curve
                    integral(i) = 0;
                elseif t == dt
                    % For t = dt, use the trapezoidal rule since there's only one interval
                    integral(i) = (dt/2) * (all_convs(1, i) + exp(-(gamma - D * k(i)^2) * t) * conv_term(i));
                else
                    num_intervals = t/dt - 1;  % Calculate number of intervals, excluding the last point
    
                    % Apply Simpson's Rule
                    simpson_integral = 0;
                    for tau = dt:dt:(num_intervals*dt)
                        t_index = int32(tau/dt);
                        f_tau = exp(-(gamma - D * k(i)^2) * tau) * all_convs(t_index, i);
    
                        if mod(t_index, 2) == 0  % even index
                            simpson_integral = simpson_integral + 2 * f_tau;
                        else  % odd index
                            simpson_integral = simpson_integral + 4 * f_tau;
                        end
                    end
    
                    % Final Simpson's Rule computation
                    integral(i) = (dt/3) * (all_convs(1, i) + simpson_integral + exp(-(gamma - D * k(i)^2) * t) * conv_term(i));
                end
            end
    
            % Crank-Nicholson update
            u_hat_lin = (u_hat_lin - 0.5 * (gamma / K) *exp((gamma - D * k.^2) * t) .* integral + ...
                         0.5 * u_hat_n1) ./ (1 + 0.5 * (gamma / K) * exp((gamma - D * k.^2) * t) .* integral);
                     
            % Check for convergence
            if max(abs(u_hat_lin - u_hat_n1)) < tol
                converged = true;
%                 msg = ['Convergence reached. Continue with t = ', num2str(t+dt), ' -------------------------'];
%                 disp(msg);
            elseif isnan(max(abs(u_hat_lin - u_hat_n1)))
                error(['NaN observed for gamma = ', num2str(gamma), ', D = ', num2str(D), ', K = ', num2str(K)]);
%                 msg = ['t = ', num2str(t), ': max(abs(u_hat_temp - u_hat_n1)) = ', num2str(max(abs(u_hat_lin - u_hat_n1)))];
%                 disp(msg);
            end
            
            % update for the next iteration
            u_hat_n1 = u_hat_lin; 
            
        end
    
        % save result of the current time step
        u_hat_alln(current_time_it + 1, :) = u_hat_n1;
        all_convs(current_time_it + 1, :) = conv_term;
                
        % Update for the next time step
        u_hat_n = u_hat_n1;
        current_time_it = current_time_it + 1;
    end
    
    % match the values in wavelengths to the values of k
    k_min_dist = zeros(size(wavelengths));  % array of k-values with the lowest distance to the entries of wavelengths
    k_min_dist_ind = zeros(size(wavelengths));  % indices
    for i = 1:length(wavelengths)
        [~, idx] = min(abs(k - wavelengths(i)));
        k_min_dist(i) = k(idx);
        k_min_dist_ind(i) = idx;
    end

    u_hat_wavelengths = u_hat_n1(end, k_min_dist_ind);
    P_raw = abs(u_hat_wavelengths).^2;
    P = P_0 * P_raw;  %/max(P_raw);
end