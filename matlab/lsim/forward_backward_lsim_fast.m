
function [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out]...
    = forward_backward_lsim_fast( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants )

%FORWARD_BACKWARD_LSIM_FAST  LSIM forward-backward with the fast (SVI) backward recursion.
%
%   Same arguments and outputs as forward_backward_lsim; fb_constants defaults to
%   lsim_fb_constants('fast') and must have variant 'fast' (lsim_backward_fast).
%   The backward message uses fixed coupling weights raised to the power
%   w_opt = sum_w( svi_eps + coupling_tetha_IM(zee,w) ); no sum(M)^2 x T arrays.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See forward_backward_lsim for the full description of the arguments.

if nargin < 6
    fb_constants = lsim_fb_constants('fast');
end

if ~strcmp(fb_constants.variant , 'fast')
    error('forward_backward_lsim_fast: fb_constants.variant must be ''fast''')
end

if nargout>1

    [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out] = ...
        forward_backward_lsim( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants );

else

    P_O_model = forward_backward_lsim( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants );

end

end
