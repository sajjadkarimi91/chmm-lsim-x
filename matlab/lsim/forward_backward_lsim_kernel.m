function [P_O_model ,alpha , beta , alpha_T , b_c_ot_nc , P_observ_cond_to_state , P_observ_cond_to_state_comp]...
    = forward_backward_lsim_kernel( pi_0_lsim , transitions_matrices , coupling_tetha_IM , P_all , mu_all  , sigma_all ,  channel_time_series , state_numbers_index , channel_num_states ,  dimension_numbers_index , num_gmm_component , sigma_diag , fb_constants )
%FORWARD_BACKWARD_LSIM_KERNEL  LSIM forward-backward on packed arrays.
%
%   [log_lik, alpha, beta, alpha_T, b_c_ot_nc, B, B_comp] = forward_backward_lsim_kernel( ...
%       pi_0, A, theta, P_all, mu_all, sigma_all, Y, state_numbers_index, M, ...
%       dimension_numbers_index, K, sigma_diag, fb_constants)
%
% The engine behind forward_backward_lsim(_sed/_fast), em_lsim(_fast/_poise) and the
% discriminative trainer. It evaluates the GMM emission densities, runs the exact LSIM
% forward pass (shared by every variant) and then the backward recursion selected by
% fb_constants.variant, which is the function lsim_backward_<variant> (see
% lsim/backward/Contents.m). Built-in variants: 'poise' (default), 'sed', 'fast'.
%
% Inputs, in the packed representation of lsim_pack_para:
%   pi_0                   sum(M) x 1 (regularized here with fb_constants.pi0_reg)
%   A                      sum(M) x sum(M); block (c,zee) is M(c) x M(zee), rows sum to 1,
%                          A(n_c, n_zee) = P(v^zee_t = n_zee | v^c_t-1 = n_c)
%   theta                  C x C coupling, sum(theta) = ones(1,C)
%   P_all                  C x max(M) x max(K)
%   mu_all                 sum(d) x max(M) x max(K)
%   sigma_all              sum(d) x max(M) x max(K)           when sigma_diag = 1
%                          sum(d) x max(d) x max(M) x max(K)  when sigma_diag = 0
%   Y                      sum(d) x T, NaN = missing
%   state_numbers_index    [0;cumsum(M)],  dimension_numbers_index  [0;cumsum(d)]
%   fb_constants           optional, see lsim_fb_constants
%
% Outputs (sum(M) x T unless noted):
%   log_lik     sum_t sum_c log P(o^c_t | o_1:t-1), the LSIM likelihood approximation
%   alpha       one-step predictive marginals, alpha(:,1) = pi_0
%   beta        backward messages, normalized so that alpha.*beta = alpha_T
%   alpha_T     smoothed marginals, each channel block sums to 1 at every t
%   b_c_ot_nc   B(:,t) divided by the channel's predictive likelihood at t
%   B, B_comp   emission densities (sum(M) x T) and per component (sum(M) x max(K) x T)

if nargin < 13 || isempty(fb_constants)
    fb_constants = lsim_fb_constants();
end

C =  ( length( channel_num_states ));
T =  ( size( channel_time_series , 2) );

P_ot_c_cond_past = zeros( C , T  );
b_c_ot_nc  = zeros( state_numbers_index(end) , T );

alpha = zeros( state_numbers_index(end) , T  );
a_alpha_b = zeros(state_numbers_index(end) , C , T-1 );


% emission probabilities
if sigma_diag
    [P_observ_cond_to_state  ,   P_observ_cond_to_state_comp ]  = ...
        gmm_pdf_fast( P_all , mu_all  , sigma_all  , channel_time_series   ,  channel_num_states ,  dimension_numbers_index , num_gmm_component  );
else
    [P_observ_cond_to_state  ,   P_observ_cond_to_state_comp ]  = ...
        gmm_pdf( P_all , mu_all  , sigma_all  , channel_time_series   ,  channel_num_states ,  dimension_numbers_index , num_gmm_component  );
end


mat_mult = zeros( state_numbers_index(end) , C  );
cuopling_repmat = zeros( state_numbers_index(end) , C  );

mat_mult_one = mat_mult;
vec_mat_mult = [];


for zee =  1:C

    zee_index = ( state_numbers_index(zee)+1:state_numbers_index(zee+1) );

    % pi0_reg <= 0 leaves pi_0 exactly as given
    if fb_constants.pi0_reg > 0
        pi_0_lsim(zee_index) = (fb_constants.pi0_reg+pi_0_lsim(zee_index))/sum(fb_constants.pi0_reg+pi_0_lsim(zee_index));
    end

    vec_mat_mult = cat(1,vec_mat_mult, ((zee-1)*state_numbers_index(end) + ( state_numbers_index(zee)+1:state_numbers_index(zee+1) ) )' );

    zee_coupling = coupling_tetha_IM(:,zee)';
    cuopling_repmat( zee_index ,: ) = repmat( zee_coupling ,channel_num_states(zee),1);

end


mat_mult_one(vec_mat_mult)=1;

alpha( :  , 1) = pi_0_lsim(:);

P_ot_c_cond_past(: , 1) = mat_mult_one'*(alpha(: , 1).* P_observ_cond_to_state(:,1));
b_c_ot_nc( : , 1) = P_observ_cond_to_state(:,1) ./ (mat_mult_one*P_ot_c_cond_past(: , 1));

%Compute Forward variables & Scaling coefficients
for t = (2:T)

    alpha_tm1_tm1 = alpha(: , t-1).*b_c_ot_nc(:,t-1);

    mat_mult(vec_mat_mult) = alpha_tm1_tm1;

    a_alpha_b( : , : ,t-1)= transitions_matrices'*mat_mult ;%compute from t=1 to t=T-1
    a_alpha_b_coupled = a_alpha_b(  : , : ,t-1).*cuopling_repmat;
    alpha(: , t) = sum( a_alpha_b_coupled , 2);

    P_ot_c_cond_past(: , t) = mat_mult_one'*(alpha(: , t).* P_observ_cond_to_state(:,t));

    emmbed_P_ot_c_cond_past = (mat_mult_one*P_ot_c_cond_past(: , t));
    b_c_ot_nc( : , t) = P_observ_cond_to_state(:,t) ./ emmbed_P_ot_c_cond_past;

end


% backward recursion: the plug-in lsim_backward_<variant>
layout = struct( 'C' , C , 'T' , T , ...
    'state_numbers_index' , state_numbers_index , ...
    'channel_num_states' , channel_num_states , ...
    'vec_mat_mult' , vec_mat_mult , ...
    'mat_mult_one' , mat_mult_one , ...
    'coupling_incoming' , cuopling_repmat );

[beta , alpha_T] = feval( ['lsim_backward_' fb_constants.variant] , ...
    alpha , b_c_ot_nc , transitions_matrices , coupling_tetha_IM , layout , fb_constants );


P_O_model = sum( log(P_ot_c_cond_past(:)) );

if(isnan(P_O_model))
    error('failed to compute log likelyhood')
end

end
