function [x, x_ws, pooler, W_in, W_hat] = evolrc(u, seed, omega_in, rho, Nh, num_pop, gen, a, ws, x0)

[Nu, time_steps] = size(u);

if nargin < 8 % no a, ws, x0
    a = 1;
    ws = 0;
    x = zeros(Nh * (num_pop)^gen,1);
elseif nargin == 8 % no ws, x0
    ws = 0;
    x = zeros(Nh * (num_pop)^gen,1);
elseif nargin == 9 % no x0
    x = zeros(Nh * (num_pop)^gen,1);
else
    x = x0;
end

if a < 0 || a > 1
    error('The parameter a must be in [0, 1]')
else
    W_hat = initStateMatrix(Nh, rho, seed, 1, a);

    for i = 1:gen
        pop = {};
        for i = 1:num_pop
            pop{end+1} = W_hat;
        end
        W_hat = combine(pop, seed);
    end

    W_in = initInputMatrix(Nu, omega_in, length(W_hat), seed, a);
    
    % Add ones for bias
    u = [u; ones(1, time_steps)];
    
    % LI-ESN
    for t=1:time_steps
        x = cat(2, x, (1-a)*x(:,end) + a*tanh(W_in*u(:,t) + W_hat*x(:,end)));
    end
    
    % Discard the initial state
    x = x(:, 2:end);
    
    % Discard the washout
    x_ws = x(:, ws+1:end);
    
    pooler = x(:, end);
end

end

