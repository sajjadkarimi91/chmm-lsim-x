

function [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out]...
    = forward_backward_lsim_sed( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants )

%FORWARD_BACKWARD_LSIM_SED  LSIM forward-backward with the SED backward recursion.
%
%   Same arguments and outputs as forward_backward_lsim; fb_constants defaults to
%   lsim_fb_constants('sed') and must have variant 'sed' (lsim_backward_sed).
%   The SED delta weights x = ( diag(f-h) + h*ones(C,C) ) \ f are evaluated in
%   closed form by the Sherman-Morrison identity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% See forward_backward_lsim for the full description of the arguments.

if nargin < 6
    fb_constants = lsim_fb_constants('sed');
end

if ~strcmp(fb_constants.variant , 'sed')
    error('forward_backward_lsim_sed: fb_constants.variant must be ''sed''')
end

if nargout>1

    [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out] = ...
        forward_backward_lsim( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants );

else

    P_O_model = forward_backward_lsim( pi_0_lsim , coupling_tetha_IM , transition_matrices_IM ,  gmm_para_lsim  ,  Y_in  , fb_constants );

end

end
