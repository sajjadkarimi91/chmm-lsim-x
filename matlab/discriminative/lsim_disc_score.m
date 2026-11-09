function L = lsim_disc_score(p, Y, fb_constants)
% Per-sequence LSIM score, L(n) = P_O_model of forward_backward_lsim_kernel.
%
% SCORE SEMANTICS. P_O_model = sum_t sum_c log sum_{i in c} alpha(i,t) p(o_t^c|i),
% where alpha is the per-channel predictive marginal propagated from
% PER-CHANNEL filtered beliefs (an assumed-density, factorized filter). For
% C = 1 this is the exact HMM log-likelihood. For C > 1 it is an approximate
% score, not the exact joint log-likelihood of the coupled model.
%
% The score is computed by the forward pass only, which is shared verbatim by
% the 'sed', 'fast' and 'poise' variants; only the backward pass differs. It is
% therefore evaluated with the 'fast' variant (same pi0_reg), whose backward
% needs no S x S x T arrays -- the value is identical to the SED one.
%
% A kernel failure or a non-finite score is returned as NaN / -Inf so the
% caller can reject the trial point instead of crashing.

fbc = fb_constants;
fbc.variant = 'fast';

if ~iscell(Y)
    Y = {Y};
end
L = zeros(numel(Y), 1);
if isfield(fbc, 'parallel') && fbc.parallel
    % extra.parallel: one kernel call per sequence, independent, so parfor gives the
    % same values as the serial loop below
    parfor n = 1:numel(Y)
        L(n) = score_one(p, Y{n}, fbc);
    end
    return
end
for n = 1:numel(Y)
    L(n) = score_one(p, Y{n}, fbc);
end
end

function v = score_one(p, Yn, fbc)
try
    v = forward_backward_lsim_kernel(p.pi_0, p.transitions_matrices, p.coupling_tetha_IM, ...
        p.P_all, p.mu_all, p.sigma_all, Yn, p.state_numbers_index, p.channel_num_states, ...
        p.dimension_numbers_index, p.num_gmm_component, p.sigma_diag, fbc);
catch
    v = NaN;
end

end
