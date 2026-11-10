

function [pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim ,  AIC, log_likelyhood, BIC, pi_steady] = em_lsim_fast( channels_observations , channel_num_states , num_gmm_component , max_itration , extra, init_model)

%EM_LSIM_FAST  EM for LSIM with the fast (SVI) E-step.
%
%   Same arguments and outputs as em_lsim, with the E-step recursion
%   lsim_backward_fast (initialization, M-step, AIC/BIC and convergence are em_lsim's).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See em_lsim for the full description of the arguments and outputs.
%
% extra.fb_constants defaults to lsim_fb_constants('fast') and may be set to
% override individual constants, e.g. lsim_fb_constants('fast','pi0_reg',1e-6).

if nargin < 5 || isempty(extra)
    extra = struct();
end

if ~isfield(extra,'fb_constants')
    extra.fb_constants = lsim_fb_constants('fast');
end

if ~strcmp(extra.fb_constants.variant , 'fast')
    error('em_lsim_fast: extra.fb_constants.variant must be ''fast''')
end

if nargin == 6
    [pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim ,  AIC, log_likelyhood, BIC, pi_steady] = ...
        em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra, init_model);
else
    [pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim ,  AIC, log_likelyhood, BIC, pi_steady] = ...
        em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);
end

end
