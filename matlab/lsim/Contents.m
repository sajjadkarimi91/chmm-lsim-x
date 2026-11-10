% LSIM  Latent structure influence models: inference and EM learning.
%
%   forward_backward_lsim       - LSIM forward-backward, cell API; recursion chosen by fb_constants.variant
%   forward_backward_lsim_sed   - forward_backward_lsim with the SED backward recursion
%   forward_backward_lsim_fast  - forward_backward_lsim with the fast (SVI) backward recursion
%   forward_backward_lsim_poise - standalone POISE forward-backward, optional refinement switch
%   forward_backward_lsim_svi   - standalone SVI forward-backward (diagonal covariance only)
%   forward_backward_lsim_kernel- shared engine on packed arrays: emissions, forward pass, backward plug-in
%   lsim_fb_constants           - variant name and numerical constants of the forward-backward
%   em_lsim                     - maximum-likelihood EM for an LSIM with GMM emissions
%   em_lsim_fast                - em_lsim with the fast (SVI) E-step
%   em_lsim_poise               - em_lsim with the POISE E-step, optional refinement switch
%   em_lsim_exact               - EM with approximate and exact (joint-state) E-steps, small models
%   viterbi_lsim                - approximate per-channel Viterbi decoding
%
% Backward recursions are plug-ins in lsim/backward (lsim_backward_<variant>).
