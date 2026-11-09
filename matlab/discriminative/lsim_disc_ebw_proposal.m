function [q, info] = lsim_disc_ebw_proposal(p, num, den, dom, damping_E, opts)
% Extended-Baum-Welch-style proposal from SIGNED expected counts, all of which
% were computed under the SAME current model p.
%
%   num, den : block counts (lsim_disc_counts) of the positively and negatively
%              weighted sequences. The update uses d = num - den. Separately
%              normalized EM estimates are never subtracted.
%   damping_E: EBW damping multiplier E. Each block uses
%                D = max(damping_E * (den mass of the block), 2 * D_min),
%              the standard EBW rule, with D_min the smallest D keeping the
%              block feasible (below).
%
% Probability block p (support only) with signed counts d:
%     p_new = (d + D p) / (sum(d) + D)
% D_min = max(0, max_j(-d_j/p_j), -sum(d)), so 2*D_min makes every supported
% entry strictly positive. p_new - p = (d - p sum d)/(sum d + D) is a positive
% multiple of the logit-coordinate gradient d - p sum d when the counts are
% exact posterior counts; its inner product with the gradient is then
% [sum d_j^2/p_j - (sum d)^2]/(sum d + D) >= 0 by Cauchy-Schwarz. That makes it
% an ascent DIRECTION, not a guaranteed ascent STEP; the caller backtracks and
% checks the measured objective.
%
% Diagonal Gaussian (per component, per feature dim, masked counts n, s, r):
%     mu_new = (s + D mu)/(n + D)
%     v_new  = (r + D (v + mu^2))/(n + D) - mu_new^2
% D_min is the largest root of v D^2 + (r + n(v+mu^2) - 2 s mu) D + (r n - s^2)
% (so v_new > 0) and -n (so n + D > 0); one D per component, as in EBW. With zero
% denominator counts D = 0 and this is the ordinary EM re-estimate.
%
% Structural zeros stay zero (they are off-support). The result is projected
% into the compact domain (probability floor, mean box, variance bounds).
%
% opts (optional): .update_blocks = subset of {'pi','A','theta','w','mu','var'}
%   (default all): families not listed are returned exactly as they are in p.
%
% Used by the legacy MMI recipe (with step search); the published recipe uses
% lsim_disc_published_update (fixed E, I-smoothing, weak-sense discrete updates).
% References: Povey & Woodland ICASSP 2001 (eqs. 3-4, the D rule); Woodland & Povey 2002.
if nargin < 6 || isempty(opts)
    opts = struct();
end

q = p;
sni = p.state_numbers_index;
D_used = [];

for z = 1:p.C
    zi = sni(z)+1:sni(z+1);

    [q.pi_0(zi), Du] = ebw_block(p.pi_0(zi), num.pi(zi), den.pi(zi), dom.support_pi(zi), damping_E);
    D_used(end+1) = Du; %#ok<AGROW>

    for i = 1:sni(end)
        [row, Du] = ebw_block(p.transitions_matrices(i, zi)', num.A(i, zi)', den.A(i, zi)', ...
            dom.support_A(i, zi)', damping_E);
        q.transitions_matrices(i, zi) = row';
        D_used(end+1) = Du; %#ok<AGROW>
    end

    [q.coupling_tetha_IM(:, z), Du] = ebw_block(p.coupling_tetha_IM(:, z), num.theta(:, z), ...
        den.theta(:, z), dom.support_theta(:, z), damping_E);
    D_used(end+1) = Du; %#ok<AGROW>

    di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
    for i = 1:p.channel_num_states(z)
        K = p.num_gmm_component(z);
        [w, Du] = ebw_block(reshape(p.P_all(z, i, 1:K), K, 1), reshape(num.occ(z, i, 1:K), K, 1), ...
            reshape(den.occ(z, i, 1:K), K, 1), reshape(dom.support_w(z, i, 1:K), K, 1), damping_E);
        q.P_all(z, i, 1:K) = w;
        D_used(end+1) = Du; %#ok<AGROW>

        for k = 1:K
            [mu, v, Du] = ebw_gauss(p.mu_all(di, i, k), p.sigma_all(di, i, k), ...
                num.n(di, i, k) - den.n(di, i, k), num.s(di, i, k) - den.s(di, i, k), ...
                num.r(di, i, k) - den.r(di, i, k), den.occ(z, i, k), damping_E);
            q.mu_all(di, i, k) = mu;
            q.sigma_all(di, i, k) = v;
            D_used(end+1) = Du; %#ok<AGROW>
        end
    end
end

q = lsim_disc_domain('project', q, dom);
if isfield(opts, 'update_blocks') && ~isempty(opts.update_blocks)
    fam = {'pi', 'pi_0'; 'A', 'transitions_matrices'; 'theta', 'coupling_tetha_IM'; ...
        'w', 'P_all'; 'mu', 'mu_all'; 'var', 'sigma_all'};
    bad = setdiff(opts.update_blocks, fam(:, 1));
    if ~isempty(bad)
        error('em_lsim_discriminative:badUpdateBlocks', 'Unknown update block(s): %s', strjoin(bad, ', '));
    end
    for j = 1:size(fam, 1)
        if ~ismember(fam{j, 1}, opts.update_blocks)
            q.(fam{j, 2}) = p.(fam{j, 2});   % frozen family: exactly the current values
        end
    end
end
info.max_D = max([D_used, 0]);
info.median_D = median(D_used);

end

function [p_new, D] = ebw_block(p, dnum, dden, support, damping_E)
p_new = p;
D = 0;
idx = support(:) & true(size(p(:)));
if nnz(idx) <= 1
    return
end
pv = p(idx);
d = dnum(idx) - dden(idx);
den_mass = sum(dden(idx));
if sum(abs(dnum(idx))) + den_mass == 0
    return  % no information about this block
end
D_min = max([0; -d./pv; -sum(d)]);
D = max(damping_E*den_mass, 2*D_min);
denom = sum(d) + D;
if ~(denom > 0)
    return
end
p_new(idx) = (d + D*pv)/denom;
end

function [mu_new, v_new, D] = ebw_gauss(mu, v, n, s, r, den_occ, damping_E)
mu_new = mu;
v_new = v;
D = 0;
if all(n == 0) && all(s == 0) && den_occ == 0
    return
end
% smallest D keeping every dim's variance positive and n + D > 0
a = v;
b = r + n.*(v + mu.^2) - 2*s.*mu;
c = r.*n - s.^2;
disc = b.^2 - 4*a.*c;
root = -inf(size(mu));
has = disc >= 0;
root(has) = (-b(has) + sqrt(disc(has)))./(2*a(has));
D_min = max([0; root(:); -n(:)]);
D = max(damping_E*den_occ, 2*D_min);
denom = n + D;
ok = denom > 1e-12*max(1, abs(D));
mu_new(ok) = (s(ok) + D*mu(ok))./denom(ok);
v_new(ok) = (r(ok) + D*(v(ok) + mu(ok).^2))./denom(ok) - mu_new(ok).^2;
bad = ~(v_new > 0) | ~isfinite(v_new) | ~isfinite(mu_new);
mu_new(bad) = mu(bad);
v_new(bad) = v(bad);
end
