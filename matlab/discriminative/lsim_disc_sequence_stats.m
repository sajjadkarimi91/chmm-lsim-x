function st = lsim_disc_sequence_stats(p, Y, fb_constants)
% Expected complete-data sufficient statistics of ONE sequence under the packed
% model p, from forward_backward_lsim_kernel with the given variant. These are
% exactly the quantities em_lsim.m accumulates in its E-step, kept per sequence
% so they can be signed and weighted before aggregation.
%
% Fields (S = sum of channel states, D = sum of feature dims):
%   loglik  the kernel score P_O_model (see lsim_disc_score for its semantics)
%   pi      S x 1       gamma at t = 1 (alpha_T(:,1))
%   xi      S x S       sum_t alpha(i,t) b(i,t) theta(c(i),z(j)) A(i,j) beta(j,t+1)
%                       -- em_lsim's sum3_zeta. For every destination channel z and
%                       every t, sum over (c, i in c, j in z) of the per-t term is
%                       exactly 1 (the kernel normalizes alpha.*beta per channel),
%                       so xi(i,j) is an expected count of
%                       "source channel c(i) chosen for z(j), s^c_t = i, s^z_{t+1} = j".
%                       Transition rows and coupling columns are its two marginals.
%   occ     C x maxS x maxK  component occupancy sum_t gamma(i,k,t) (unmasked)
%   n, s, r D x maxS x maxK  masked occupancy, first and elementwise second moment
%                       per feature dim: missing entries (NaN) contribute to none.
%
% For C = 1 the kernel is the exact HMM forward-backward, so these are exact
% posterior expectations and, by Fisher's identity, give the exact gradient of
% the score. For C > 1 they come from an approximate smoother and are only a
% SURROGATE direction for the score (see tests/check_disc_gradients.m).

S = p.state_numbers_index(end);
C = p.C;
T = size(Y, 2);
maxS = max(p.channel_num_states);
maxK = max(p.num_gmm_component);
D = p.dimension_numbers_index(end);

[ll, alpha, beta, alpha_T, b, Pst, Pcomp] = forward_backward_lsim_kernel(p.pi_0, ...
    p.transitions_matrices, p.coupling_tetha_IM, p.P_all, p.mu_all, p.sigma_all, Y, ...
    p.state_numbers_index, p.channel_num_states, p.dimension_numbers_index, ...
    p.num_gmm_component, p.sigma_diag, fb_constants);

Pst = reshape(Pst, S, T);
Pcomp = reshape(Pcomp, S, maxK, T);

st.loglik = ll;
st.pi = alpha_T(:, 1);

% xi, exactly as em_lsim builds coupling_transition_all and sum3_zeta
membership = zeros(S, C);
for c = 1:C
    membership(p.state_numbers_index(c)+1:p.state_numbers_index(c+1), c) = 1;
end
coupling_transition_all = zeros(S, S);
for z = 1:C
    zi = p.state_numbers_index(z)+1:p.state_numbers_index(z+1);
    coupling_transition_all(:, zi) = repmat(membership*p.coupling_tetha_IM(:, z), 1, numel(zi)) ...
        .* p.transitions_matrices(:, zi);
end
if T > 1
    filtered = alpha(:, 1:T-1).*b(:, 1:T-1);
    st.xi = (filtered*beta(:, 2:T)') .* coupling_transition_all;
else
    st.xi = zeros(S, S);
end

st.occ = zeros(C, maxS, maxK);
st.n = zeros(D, maxS, maxK);
st.s = zeros(D, maxS, maxK);
st.r = zeros(D, maxS, maxK);

for z = 1:C
    di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
    Yz = Y(di, :);
    mask = ~isnan(Yz);
    Y0 = Yz;
    Y0(~mask) = 0;
    for i = 1:p.channel_num_states(z)
        row = p.state_numbers_index(z) + i;
        for k = 1:p.num_gmm_component(z)
            g = alpha_T(row, :) .* (reshape(Pcomp(row, k, :), 1, T) ./ Pst(row, :));
            g(isnan(g)) = 0;
            st.occ(z, i, k) = sum(g);
            st.n(di, i, k) = double(mask)*g';
            st.s(di, i, k) = Y0*g';
            st.r(di, i, k) = (Y0.^2)*g';
        end
    end
end

fields = {'pi', 'xi', 'occ', 'n', 's', 'r'};
for f = 1:numel(fields)
    if any(~isfinite(st.(fields{f})(:)))
        error('em_lsim_discriminative:nonfiniteStats', ...
            'Non-finite %s statistics (score %g); the model assigns this sequence zero mass.', ...
            fields{f}, ll);
    end
end

end
