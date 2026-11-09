function [ll, st] = lsim_disc_exact_oracle(p, Y)
% EXACT inference for a tiny LSIM by enumerating the joint (Cartesian) state.
% A reference only -- exponential in C -- never a training path.
%
% The generative model (the toolbox's LSIM semantics, as in im_para_eqhmm):
%   s_1^c ~ pi^c independently;
%   for each destination channel z at t+1, a source channel u ~ theta(:, z) is
%   chosen and s_{t+1}^z ~ A{u,z}(s_t^u, :), independently over z given s_t,
%   so P(s_{t+1} | s_t) = prod_z sum_c theta(c,z) A{c,z}(s_t^c, s_{t+1}^z);
%   o_t^c ~ GMM of state s_t^c, independently over channels given s_t.
% pi_0 is used exactly as given (call the kernel with pi0_reg = 0 to compare).
%
% Returns the exact log-likelihood and the exact expected statistics in the
% layout of lsim_disc_sequence_stats (pi, xi, occ, n, s, r), where xi(i,j) is
% the exact E[#{t : u_{t+1}^{z(j)} = c(i), s_t^{c(i)} = i, s_{t+1}^{z(j)} = j}],
% i.e. the latent source-selection responsibility the coupling update needs.

C = p.C;
Sc = p.channel_num_states(:)';
sni = p.state_numbers_index;
S = sni(end);
T = size(Y, 2);
maxS = max(Sc);
maxK = max(p.num_gmm_component);
D = p.dimension_numbers_index(end);

% joint state enumeration: idx(c, j) = state of channel c in joint state j
J = prod(Sc);
idx = zeros(C, J);
for j = 1:J
    rem_j = j - 1;
    for c = C:-1:1
        idx(c, j) = mod(rem_j, Sc(c)) + 1;
        rem_j = floor(rem_j/Sc(c));
    end
end
gidx = idx + sni(1:C);   % global state index per channel

% emissions (the kernel's own emission code)
[Pst, Pcomp] = gmm_pdf_fast(p.P_all, p.mu_all, p.sigma_all, Y, p.channel_num_states, ...
    p.dimension_numbers_index, p.num_gmm_component);
Pst = reshape(Pst, S, T);
Pcomp = reshape(Pcomp, S, maxK, T);

E = ones(J, T);
for c = 1:C
    E = E .* Pst(gidx(c, :), :);
end
p0 = ones(J, 1);
for c = 1:C
    p0 = p0 .* p.pi_0(gidx(c, :));
end
% joint transition and the per-(z, j, prev) source mixture
Tr = ones(J, J);
mix = cell(C, 1);    % mix{z}(a, b) = sum_c theta(c,z) A(s_a^c, s_b^z)
for z = 1:C
    mz = zeros(J, J);
    for c = 1:C
        mz = mz + p.coupling_tetha_IM(c, z) * p.transitions_matrices(gidx(c, :), gidx(z, :));
    end
    mix{z} = mz;
    Tr = Tr .* mz;
end

% scaled forward-backward on the joint chain
alpha = zeros(J, T);
scale = zeros(1, T);
a = p0 .* E(:, 1);
scale(1) = sum(a);
alpha(:, 1) = a/scale(1);
for t = 2:T
    a = (Tr' * alpha(:, t-1)) .* E(:, t);
    scale(t) = sum(a);
    alpha(:, t) = a/scale(t);
end
ll = sum(log(scale));
beta = ones(J, T);
for t = T-1:-1:1
    beta(:, t) = Tr * (E(:, t+1).*beta(:, t+1)) / scale(t+1);
end
gamma = alpha.*beta;

st.loglik = ll;
st.pi = zeros(S, 1);
for c = 1:C
    for i = 1:Sc(c)
        st.pi(sni(c)+i) = sum(gamma(idx(c, :) == i, 1));
    end
end

st.xi = zeros(S, S);
for t = 1:T-1
    xi_t = (alpha(:, t) * (E(:, t+1).*beta(:, t+1))') .* Tr / scale(t+1);   % J x J
    for z = 1:C
        for c = 1:C
            % responsibility of source c for destination z, per (prev, next)
            resp = (p.coupling_tetha_IM(c, z) * p.transitions_matrices(gidx(c, :), gidx(z, :))) ./ mix{z};
            resp(mix{z} == 0) = 0;
            w = xi_t .* resp;
            for i = 1:Sc(c)
                for jj = 1:Sc(z)
                    st.xi(sni(c)+i, sni(z)+jj) = st.xi(sni(c)+i, sni(z)+jj) + ...
                        sum(sum(w(idx(c, :) == i, idx(z, :) == jj)));
                end
            end
        end
    end
end

st.occ = zeros(C, maxS, maxK);
st.n = zeros(D, maxS, maxK);
st.s = zeros(D, maxS, maxK);
st.r = zeros(D, maxS, maxK);
for c = 1:C
    di = p.dimension_numbers_index(c)+1:p.dimension_numbers_index(c+1);
    Yc = Y(di, :);
    mask = ~isnan(Yc);
    Y0 = Yc;
    Y0(~mask) = 0;
    for i = 1:Sc(c)
        g_ci = sum(gamma(idx(c, :) == i, :), 1);
        row = sni(c) + i;
        for k = 1:p.num_gmm_component(c)
            g = g_ci .* reshape(Pcomp(row, k, :), 1, T) ./ Pst(row, :);
            g(isnan(g)) = 0;
            st.occ(c, i, k) = sum(g);
            st.n(di, i, k) = double(mask)*g';
            st.s(di, i, k) = Y0*g';
            st.r(di, i, k) = (Y0.^2)*g';
        end
    end
end

end
