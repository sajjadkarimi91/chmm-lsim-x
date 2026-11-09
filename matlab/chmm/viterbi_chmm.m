function [P_star_model, X_star] = viterbi_chmm(varargin)
%VITERBI_CHMM  Most likely hidden-state sequence.
%
%   [log_prob, X_star] = viterbi_chmm(pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, Y)
%       EXACT Viterbi decoding of a full CHMM (the parameters of em_chmm), on the joint
%       state space of prod(M) states. log_prob is the log-probability of the best joint
%       path together with the observations; X_star is C x T (states numbered from 1).
%
%   [log_score, X_star] = viterbi_chmm(pi_0_lsim, coupling_tetha_IM, transition_matrices_IM, gmm_para_lsim, Y)
%       approximate per-channel decoding of an LSIM, identical to viterbi_lsim (this
%       calling form is kept for existing code).
%
% Y is a C x 1 cell, Y{c} is d(c) x T (NaN = missing). Emissions are Gaussian mixtures
% with diagonal (d x 1) or full (d x d) covariances. Python twin: chmm_lsim.viterbi_chmm.

if nargin == 5
    [P_star_model, X_star] = viterbi_lsim(varargin{:});
    return
end
if nargin ~= 4
    error('viterbi_chmm:nargin', ['use viterbi_chmm(pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, Y) ' ...
        'for a CHMM or viterbi_lsim(pi_0, coupling, transitions, gmm_para, Y) for an LSIM']);
end
[pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, Y] = varargin{:};

[pi_joint, transition_joint, log_B, index_matrix] = ...
    chmm_joint_model(pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, Y);
[J, T] = size(log_B);

log_A = log(transition_joint);            % J x J, rows: previous joint state
delta = log(pi_joint) + log_B(:, 1);
psi = zeros(J, T);
for t = 2:T
    [best, arg] = max(delta + log_A, [], 1);
    delta = best(:) + log_B(:, t);
    psi(:, t) = arg(:);
end

path = zeros(1, T);
[P_star_model, path(T)] = max(delta);
for t = T-1:-1:1
    path(t) = psi(path(t+1), t+1);
end
X_star = index_matrix(:, path);

end
