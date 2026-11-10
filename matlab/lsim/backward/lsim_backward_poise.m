function [beta , alpha_T] = lsim_backward_poise( alpha , b_c_ot_nc , transitions_matrices , coupling_tetha_IM , layout , fb_constants )
%LSIM_BACKWARD_POISE  POISE backward recursion of the LSIM forward-backward ('poise').
%
%   [beta, alpha_T] = lsim_backward_poise(alpha, b_c_ot_nc, A, theta, layout, fb_constants)
%
% Power-corrected outgoing-influence recursion (power outside the mixture). For
% channel xi with theta_out = sum_c theta(xi,c) and rho_c = theta(xi,c)/theta_out,
%   q_t(m)    = sum_c rho_c sum_n A{xi,c}(m,n) beta^c_t+1(n)
%   beta_t(m) proportional to b_t(m) q_t(m)^theta_out
% (beta_t = b_t for a channel with theta_out = 0).
%
% With fb_constants.poise_refinement = true the raw smoothed marginals are then
% refined once, from the frozen raw marginals:
%   alpha_T(:,t) = w1 * propagated(t) + (1-w1) * alpha_T_raw(:,t),
%   w1 = max(0, 0.75-theta_out),
% where propagated(t) is alpha_T_raw(:,t-1) pushed through the coupled transitions
% and the observation at t; beta is then reset to alpha_T./alpha. The refinement
% only changes channels with theta_out < 0.75.
%
% A nonpositive or nonfinite channel normalizer raises the error
% 'forward_backward_lsim_kernel:poiseNormalizer'.
%
% Arguments and outputs follow the backward plug-in interface, see
% lsim/backward/Contents.m.

C = layout.C;
T = layout.T;
state_numbers_index = layout.state_numbers_index;
channel_num_states = layout.channel_num_states;
vec_mat_mult = layout.vec_mat_mult;
mat_mult_one = layout.mat_mult_one;

alpha_T  = zeros( state_numbers_index(end) , T  );
beta  = zeros( state_numbers_index(end) , T );
mat_mult = zeros( state_numbers_index(end) , C  );

weights_subsystems_poise = zeros( state_numbers_index(end) , C  );
theta_out_poise = zeros( state_numbers_index(end) , 1 );

for zee =  1:C

    zee_index = ( state_numbers_index(zee)+1:state_numbers_index(zee+1) );

    theta_out_channel = sum(coupling_tetha_IM(zee,:));
    theta_out_poise(zee_index) = theta_out_channel;
    if theta_out_channel > 0
        zee_coupling = coupling_tetha_IM(zee,:)/theta_out_channel;
        weights_subsystems_poise( zee_index ,: ) = ...
            repmat( zee_coupling ,channel_num_states(zee),1);
    end

end


beta(: , T) = b_c_ot_nc(: , T);
alpha_T(: , T) =  alpha(: , T).*beta(: , T);

% raw POISE backward recursion: power outside the outgoing mixture
for t = (T-1:-1:1)

    mat_mult(vec_mat_mult) = beta(: , t+1);
    beta_C_poise = transitions_matrices * mat_mult;
    q_poise = sum(beta_C_poise.*weights_subsystems_poise, 2);

    beta(:,t) = b_c_ot_nc(:,t).*(q_poise.^theta_out_poise);
    zero_out = theta_out_poise == 0;
    beta(zero_out,t) = b_c_ot_nc(zero_out,t);

    alpha_T(:,t) = beta(:,t).*alpha(:,t);
    temp_sum = cumsum(alpha_T(:,t));
    normalizer = temp_sum(state_numbers_index(2:end)) - ...
        [0;temp_sum(state_numbers_index(2:end-1))];
    check_poise_normalizer(normalizer, t, 'backward recursion');
    normalizer_vec = mat_mult_one*normalizer(:);
    beta(:,t) = beta(:,t)./normalizer_vec;
    alpha_T(:,t) = alpha_T(:,t)./normalizer_vec;

end

if fb_constants.poise_refinement && T > 1
    [beta , alpha_T] = poise_refinement( alpha , beta , alpha_T , b_c_ot_nc , ...
        transitions_matrices , theta_out_poise , layout );
end

end


function [beta , alpha_T] = poise_refinement( alpha , beta , alpha_T , b_c_ot_nc , transitions_matrices , theta_out_poise , layout )

% One-shot observation-conditioned refinement: every propagated marginal is formed
% from the frozen raw POISE marginals, never from an already refined slice.

T = layout.T;
state_numbers_index = layout.state_numbers_index;
vec_mat_mult = layout.vec_mat_mult;
mat_mult_one = layout.mat_mult_one;
cuopling_repmat = layout.coupling_incoming;

mat_mult = zeros( state_numbers_index(end) , layout.C  );

alpha_T_raw = alpha_T;
w1_poise = max(0, 0.75-theta_out_poise);

for t = 2:T
    mat_mult(vec_mat_mult) = alpha_T_raw(:,t-1);
    propagated_C = transitions_matrices'*mat_mult;
    propagated = b_c_ot_nc(:,t).*sum(propagated_C.*cuopling_repmat,2);

    temp_sum = cumsum(propagated);
    normalizer = temp_sum(state_numbers_index(2:end)) - ...
        [0;temp_sum(state_numbers_index(2:end-1))];
    check_poise_normalizer(normalizer, t, 'refinement');
    propagated = propagated./(mat_mult_one*normalizer(:));

    alpha_T(:,t) = w1_poise.*propagated + ...
        (1-w1_poise).*alpha_T_raw(:,t);
end

% effective backward message whose one-slice marginal is the refined alpha_T
% (used by the EM transition statistic)
beta(:) = 0;
positive_alpha = alpha > 0;
beta(positive_alpha) = alpha_T(positive_alpha)./alpha(positive_alpha);

end


function check_poise_normalizer(normalizer, t, stage)

% a channel's total smoothed mass must be finite and positive

bad = find(~isfinite(normalizer) | normalizer <= 0, 1);
if ~isempty(bad)
    error('forward_backward_lsim_kernel:poiseNormalizer', ...
        'POISE %s: channel %d normalizer at t=%d is nonpositive or nonfinite.', ...
        stage, bad, t);
end

end
