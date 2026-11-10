% BACKWARD  LSIM backward recursions (plug-ins of forward_backward_lsim_kernel).
%
%   lsim_backward_poise - POISE: power-corrected outgoing-influence recursion (+ optional refinement)
%   lsim_backward_sed   - SED: pairwise-conditional message with solved delta weights
%   lsim_backward_fast  - fast (SVI): fixed coupling weights raised to the power w_opt
%
% Adding a recursion: write lsim_backward_<name>.m anywhere on the path and select it
% with fbc = lsim_fb_constants('<name>') (extra.fb_constants for em_lsim and
% em_lsim_discriminative). The kernel calls
%
%   [beta, alpha_T] = lsim_backward_<name>(alpha, b_c_ot_nc, A, theta, layout, fb_constants)
%
% after the exact forward pass, with (N = sum(M) stacked states, T samples)
%   alpha        N x T  one-step predictive marginals P(v^c_t | o_1:t-1), alpha(:,1) = pi_0
%   b_c_ot_nc    N x T  emission density divided by the channel's predictive likelihood,
%                       so alpha.*b_c_ot_nc are the filtered marginals
%   A            N x N  packed transitions, block (c,zee) = P(v^zee_t | v^c_t-1)
%   theta        C x C  coupling, sum(theta) = ones(1,C)
%   layout       struct: .C, .T, .state_numbers_index ([0;cumsum(M)]),
%                .channel_num_states (M), .mat_mult_one (N x C, 1 where state i belongs
%                to channel c), .vec_mat_mult (linear indices of those ones),
%                .coupling_incoming (N x C, row i of channel zee = theta(:,zee)')
%   fb_constants the struct from lsim_fb_constants (extra fields are passed through)
% and must return
%   beta         N x T  backward messages
%   alpha_T      N x T  smoothed marginals; every channel block sums to 1 at every t
% em_lsim uses alpha_T for the state statistics and alpha.*b_c_ot_nc, beta(:,t+1)
% for the transition/coupling statistics, so beta must satisfy
% alpha_T = alpha.*beta (up to the per-channel normalization).
