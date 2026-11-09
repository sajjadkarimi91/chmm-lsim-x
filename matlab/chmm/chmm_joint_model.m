function [pi_joint, transition_joint, log_B_joint, index_matrix, B, B_comp, sigma_diag] = chmm_joint_model(pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, Y)
%CHMM_JOINT_MODEL  Joint-state (equivalent single-chain HMM) form of a full CHMM.
%
%   [pi_joint, transition_joint, log_B_joint, index_matrix, B, B_comp, sigma_diag] = ...
%       chmm_joint_model(pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, Y)
%
% Joint state j is the tuple index_matrix(:, j) of channel states; the last channel varies
% fastest (the order of chmm_cartesian_product and of kron).
%   pi_joint          J x 1, the product of the channels' initial distributions
%   transition_joint  J x J, transition_joint(j, j') = prod_z transition_matrices_chmm{z}(j, m_z(j'))
%   log_B_joint       J x T, sum over channels of the log emission density of each channel's state
%   B, B_comp         sum(M) x T and sum(M) x max(K) x T per-channel state and component
%                     densities (gmm_pdf_fast for diagonal, gmm_pdf for full covariance;
%                     missing dimensions are marginalized, a fully missing sample gives 1)
% Y is a C x 1 cell, Y{c} is d(c) x T (NaN = missing).

C = size(Y, 1);
M = cellfun(@numel, pi_0_chmm(:));
dim_observation = cellfun(@(y) size(y, 1), Y(:));
K = cellfun(@(g) numel(g.gmm_para(1).P), chmm_gmm_para(:));
J = prod(M);
state_numbers_index = [0; cumsum(M)];
dimension_numbers_index = [0; cumsum(dim_observation)];

index_matrix = zeros(C, J);
for c = 1:C
    index_matrix(c, :) = repmat(kron(1:M(c), ones(1, prod(M(c+1:end)))), 1, prod(M(1:c-1)));
end
pi_joint = 1;
for c = 1:C
    pi_joint = kron(pi_joint, pi_0_chmm{c}(:));
end
transition_joint = ones(J, J);
for z = 1:C
    transition_joint = transition_joint .* transition_matrices_chmm{z}(:, index_matrix(z, :));
end

% diagonal vs full covariance, decided on a channel with more than one dimension
zee_temp = find(dim_observation > 1, 1);
if isempty(zee_temp)
    zee_temp = 1;
end
sigma_diag = size(chmm_gmm_para{zee_temp}.gmm_para(1).sigma(1).x, 2) == 1;

P_all = zeros(C, max(M), max(K));
mu_all = zeros(sum(dim_observation), max(M), max(K));
if sigma_diag
    sigma_all = ones(sum(dim_observation), max(M), max(K));
else
    sigma_all = ones(sum(dim_observation), max(dim_observation), max(M), max(K));
end
for zee = 1:C
    dim_zee = dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1);
    gmm_para = chmm_gmm_para{zee}.gmm_para;
    for i = 1:M(zee)
        for k = 1:K(zee)
            P_all(zee, i, k) = gmm_para(i).P(k);
            mu_all(dim_zee, i, k) = gmm_para(i).mu(k).x;
            if sigma_diag
                sigma_all(dim_zee, i, k) = gmm_para(i).sigma(k).x;
            else
                sigma_all(dim_zee, 1:dim_observation(zee), i, k) = gmm_para(i).sigma(k).x;
            end
        end
    end
end
Y_all = cat(1, Y{:});
if sigma_diag
    [B, B_comp] = gmm_pdf_fast(P_all, mu_all, sigma_all, Y_all, M, dimension_numbers_index, K);
else
    [B, B_comp] = gmm_pdf(P_all, mu_all, sigma_all, Y_all, M, dimension_numbers_index, K);
end

log_B_joint = zeros(J, size(Y_all, 2));
for c = 1:C
    log_B_joint = log_B_joint + log(B(state_numbers_index(c) + index_matrix(c, :), :));
end
end
