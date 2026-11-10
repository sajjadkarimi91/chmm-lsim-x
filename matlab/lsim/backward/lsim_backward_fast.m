function [beta , alpha_T] = lsim_backward_fast( alpha , b_c_ot_nc , transitions_matrices , coupling_tetha_IM , layout , fb_constants )
%LSIM_BACKWARD_FAST  Fast (SVI) backward recursion of the LSIM forward-backward ('fast').
%
%   [beta, alpha_T] = lsim_backward_fast(alpha, b_c_ot_nc, A, theta, layout, fb_constants)
%
% For channel zee the child messages A{zee,w} * beta^w_t+1 are mixed with the fixed
% outgoing weights theta(zee,w)/(svi_eps + sum_w theta(zee,w)) and the mixture is
% raised to the power w_opt = sum_w( svi_eps + theta(zee,w) ), with
% svi_eps = fb_constants.svi_eps. No pairwise arrays are stored.
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

weights_subsystems_svi = zeros( state_numbers_index(end) , C  );
cuopling_repmat_pow = zeros( state_numbers_index(end) , C  );

for zee =  1:C

    zee_index = ( state_numbers_index(zee)+1:state_numbers_index(zee+1) );

    zee_coupling = coupling_tetha_IM(zee,:)/(fb_constants.svi_eps+sum(coupling_tetha_IM(zee,:)));
    weights_subsystems_svi( zee_index ,: ) = repmat( zee_coupling ,channel_num_states(zee),1);

    zee_coupling = fb_constants.svi_eps+coupling_tetha_IM(zee,:);
    cuopling_repmat_pow( zee_index ,: ) = repmat( zee_coupling ,channel_num_states(zee),1);

end

% only active if coupling_tetha_IM itself contains NaN (the denominator above
% carries the svi_eps guard)
weights_subsystems_svi(isnan(weights_subsystems_svi)) = 1/C;

w_opt_svi = sum(cuopling_repmat_pow,2);


beta(: , T) = b_c_ot_nc(: , T);
alpha_T(: , T) =  alpha(: , T).*beta(: , T);

%Compute Backward variables
for t = (T-1:-1:1)

    mat_mult(vec_mat_mult) = beta(: , t+1);
    beta_C_svi = transitions_matrices *mat_mult ;%compute from t=1 to t=T-1

    beta(: , t) = b_c_ot_nc( : , t).*(  sum( beta_C_svi.*weights_subsystems_svi , 2) ).^w_opt_svi;

    beta(isnan(beta(: , t)) , t)=  b_c_ot_nc( isnan(beta(: , t)) , t);

    % alpha_T is formed from the normalized beta (not divided by the normalizer)
    alpha_T_temp = beta(: , t).*alpha(: , t);
    temp_sum = cumsum(alpha_T_temp);
    normalizer = temp_sum(state_numbers_index(2:end))-[0;temp_sum(state_numbers_index(2:end-1))];
    normalizer_vec = mat_mult_one*normalizer(:);
    beta( : , t) = beta( : , t)./normalizer_vec;

    alpha_T( : , t) = beta(: , t).*alpha(: , t);

end

end
