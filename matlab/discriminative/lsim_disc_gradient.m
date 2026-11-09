function g = lsim_disc_gradient(p, bc, dom)
% Unconstrained-coordinate gradient implied by signed block counts bc
% (lsim_disc_counts), in the coordinates the finite-difference checks perturb:
%
%   probability block with softmax logits u (support only):  d - p*sum(d)
%   diagonal Gaussian mean:                                  (s - n.*mu)./v
%   diagonal Gaussian log variance (v = exp(a)):              0.5*((r - 2 mu.*s + n.*mu.^2)./v - n)
%
% For exact posterior counts this is the exact gradient of the log-likelihood
% (Fisher's identity). For approximate-smoother counts it is the surrogate
% direction the trainer uses. Off-support entries get gradient 0.
%
% Fields mirror the parameter arrays: pi (S x 1), A (S x S), theta (C x C),
% w (C x maxS x maxK), mu and logv (D x maxS x maxK).

sni = p.state_numbers_index;
g.pi = zeros(size(p.pi_0));
g.A = zeros(size(p.transitions_matrices));
g.theta = zeros(size(p.coupling_tetha_IM));
g.w = zeros(size(p.P_all));
g.mu = zeros(size(p.mu_all));
g.logv = zeros(size(p.sigma_all));

for z = 1:p.C
    zi = sni(z)+1:sni(z+1);
    g.pi(zi) = logit_grad(p.pi_0(zi), bc.pi(zi), dom.support_pi(zi));
    for i = 1:sni(end)
        g.A(i, zi) = logit_grad(p.transitions_matrices(i, zi), bc.A(i, zi), dom.support_A(i, zi));
    end
    g.theta(:, z) = logit_grad(p.coupling_tetha_IM(:, z), bc.theta(:, z), dom.support_theta(:, z));
    di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
    for i = 1:p.channel_num_states(z)
        K = p.num_gmm_component(z);
        g.w(z, i, 1:K) = logit_grad(p.P_all(z, i, 1:K), bc.occ(z, i, 1:K), dom.support_w(z, i, 1:K));
        for k = 1:K
            mu = p.mu_all(di, i, k);
            v = p.sigma_all(di, i, k);
            n = bc.n(di, i, k);
            s = bc.s(di, i, k);
            r = bc.r(di, i, k);
            g.mu(di, i, k) = (s - n.*mu)./v;
            g.logv(di, i, k) = 0.5*((r - 2*mu.*s + n.*mu.^2)./v - n);
        end
    end
end

end

function gr = logit_grad(pv, d, support)
shape = size(pv);
pv = pv(:);
d = d(:);
support = support(:);
gr = zeros(size(pv));
gr(support) = d(support) - pv(support)*sum(d(support));
gr = reshape(gr, shape);
end
