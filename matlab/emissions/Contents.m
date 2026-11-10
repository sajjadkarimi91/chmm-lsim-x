% EMISSIONS  Emission models: per-channel observation densities of the hidden states.
%
%   gmm_pdf_fast - Gaussian-mixture densities, diagonal covariance (vectorized)
%   gmm_pdf      - Gaussian-mixture densities, full covariance
%   gmm_gen      - draw one observation from the Gaussian mixture of a state
%
% An emission model provides, for every channel c, hidden state i of c and mixture
% component k, the density of the channel's observation y^c_t:
%   B_comp(i,k,t) = P(k | i) p(y^c_t | i, k)      (sum(M) x max(K) x T)
%   B(i,t)        = sum_k B_comp(i,k,t)           (sum(M) x T)
% with states stacked channel by channel ([0;cumsum(M)] offsets) and NaN observations
% left out (a fully missing channel has density 1). The inference code
% (forward_backward_lsim_kernel, viterbi_lsim, chmm_joint_model) only consumes B and
% B_comp. The GMM M-step (means, covariances, mixture weights) is currently
% implemented inline in em_lsim and em_chmm.
