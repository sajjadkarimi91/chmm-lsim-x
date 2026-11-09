function [log_lik, gamma, alpha, xi_sum] = chmm_exact_forward_backward(pi_joint, transition_joint, log_B)
%CHMM_EXACT_FORWARD_BACKWARD  Exact scaled forward-backward on the joint-state chain of a CHMM.
%
%   [log_lik, gamma, alpha, xi_sum] = chmm_exact_forward_backward(pi_joint, transition_joint, log_B)
%
% pi_joint (J x 1), transition_joint (J x J, rows = previous state) and log_B (J x T log
% emission densities), e.g. from chmm_joint_model. Returns the exact log-likelihood, the
% smoothed posteriors gamma (J x T), the one-step predictive distributions alpha (J x T,
% alpha(:,1) = pi_joint, as in the LSIM forward-backward functions) and, if requested, the
% summed pairwise posteriors xi_sum(j, j') = sum_t P(x_t = j, x_{t+1} = j' | data).

[J, T] = size(log_B);
shift = max(log_B, [], 1);
B = exp(log_B - shift);                    % rescaled per time step; the shift returns in log_lik

alpha = zeros(J, T);
filtered = zeros(J, T);
c = zeros(1, T);
alpha(:, 1) = pi_joint(:);
for t = 1:T
    if t > 1
        alpha(:, t) = transition_joint' * filtered(:, t-1);
    end
    a = alpha(:, t) .* B(:, t);
    c(t) = sum(a);
    if ~(c(t) > 0) || ~isfinite(c(t))
        error('chmm_exact_forward_backward:zeroLikelihood', ...
            'no joint state can explain the observation at t = %d', t);
    end
    filtered(:, t) = a / c(t);
end

beta = ones(J, T);
for t = T-1:-1:1
    beta(:, t) = transition_joint * (B(:, t+1) .* beta(:, t+1)) / c(t+1);
end
gamma = filtered .* beta;
gamma = gamma ./ sum(gamma, 1);
log_lik = sum(log(c)) + sum(shift);

if nargout > 3
    xi_sum = zeros(J, J);
    for t = 1:T-1
        xi_sum = xi_sum + (filtered(:, t) * (B(:, t+1) .* beta(:, t+1))') / c(t+1);
    end
    xi_sum = xi_sum .* transition_joint;
end
end
