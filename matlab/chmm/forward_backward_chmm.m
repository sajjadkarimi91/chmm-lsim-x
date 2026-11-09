function [P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out]...
    = forward_backward_chmm(pi_0_chmm, transition_matrices_chmm, chmm_gmm_para,  Y_in)
%FORWARD_BACKWARD_CHMM  Exact forward-backward inference for a full CHMM.
%
%   [log_lik, alpha, beta, alpha_T, b_c_ot_nc, P_observ_cond_to_state, P_observ_cond_to_state_comp] = ...
%       forward_backward_chmm(pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, Y)
%
% The parameters are those of em_chmm: pi_0_chmm{c} (M(c) x 1), transition_matrices_chmm{c}
% (prod(M) x M(c), row j = P(x^c_t | joint state j at t-1)) and chmm_gmm_para{c} (Gaussian
% mixtures, diagonal or full covariance). Y is a C x 1 cell, Y{c} is d(c) x T (NaN = missing).
% Inference is exact, on the joint state space of prod(M) states (chmm_joint_model,
% chmm_exact_forward_backward), so it is meant for small models.
%
% Outputs (cells are C x 1, one per channel, as forward_backward_lsim):
%   log_lik      exact log-likelihood of Y
%   alpha{c}     one-step predictive marginals P(x^c_t | y_1..y_{t-1}), M(c) x T
%   alpha_T{c}   smoothed marginals P(x^c_t | y_1..y_T), M(c) x T
%   beta{c}      alpha_T{c} ./ alpha{c}
%   b_c_ot_nc{c} zeros (kept for the output layout of forward_backward_lsim)
%   P_observ_cond_to_state{c}, P_observ_cond_to_state_comp{c}   state and component
%                emission densities of channel c

C = size(Y_in , 1);
[pi_joint, transition_joint, log_B_joint, index_matrix, B, B_comp] = ...
    chmm_joint_model(pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, Y_in);
[P_O_model, gamma_joint, alpha_joint] = chmm_exact_forward_backward(pi_joint, transition_joint, log_B_joint);

if nargout > 1
    M = cellfun(@numel, pi_0_chmm(:));
    K = cellfun(@(g) numel(g.gmm_para(1).P), chmm_gmm_para(:));
    state_numbers_index = [0; cumsum(M)];
    T = size(log_B_joint, 2);
    alpha_out = cell(C,1);
    beta_out = cell(C,1);
    alpha_T_out = cell(C,1);
    b_c_ot_nc_out = cell(C,1);
    P_observ_cond_to_state_out = cell(C,1);
    P_observ_cond_to_state_comp_out = cell(C,1);
    for c = 1:C
        alpha_c = zeros(M(c), T);
        alpha_T_c = zeros(M(c), T);
        for s = 1:M(c)
            alpha_c(s, :) = sum(alpha_joint(index_matrix(c,:)==s, :), 1);
            alpha_T_c(s, :) = sum(gamma_joint(index_matrix(c,:)==s, :), 1);
        end
        c_index = state_numbers_index(c)+1:state_numbers_index(c+1);
        alpha_out{c,1} = alpha_c;
        alpha_T_out{c,1} = alpha_T_c;
        beta_out{c,1} = alpha_T_c./(10^-12+alpha_c);
        b_c_ot_nc_out{c,1} = zeros(M(c), T);
        P_observ_cond_to_state_out{c,1} = B(c_index, :);
        P_observ_cond_to_state_comp_out{c,1} = B_comp(c_index, 1:K(c), :);
    end
end
end
