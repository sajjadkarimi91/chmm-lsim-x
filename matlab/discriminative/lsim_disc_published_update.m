function [q, info] = lsim_disc_published_update(p, num, den, dom, opts)
%LSIM_DISC_PUBLISHED_UPDATE One published-recipe MMI update of one model.
%
%   [q, info] = lsim_disc_published_update(p, num, den, dom, opts)
%
% p        : packed current model; num / den : UNCANCELED numerator / denominator
%            block counts (lsim_disc_counts), both computed under p.
% dom      : lsim_disc_domain struct (supports and bounds).
% opts     : .ebw_E (2), .ismooth_tau (100, frame units), .ismooth_backoff ('ml' |
%            'previous'), .ismooth_tau_discrete (0), .weak_sense_iters (100),
%            .weak_sense_tol (1e-10), .update_blocks ({} = all).
%
% Published recipe (Povey & Woodland ICASSP 2001 eqs. 3-4; Woodland & Povey 2002):
%  * I-smoothing of the Gaussian numerator statistics with tau back-off pseudo-counts:
%      gamma^num += tau, x^num += tau mu_b, S^num += tau (var_b + mu_b^2),
%    back-off 'ml' = the ML estimate from the numerator statistics (the current
%    parameters where the numerator occupancy is zero), 'previous' = the current model.
%  * Gaussian EBW update, per component and dimension,
%      mu  = (x^num - x^den + D mu0) / (gamma^num - gamma^den + D)
%      var = (S^num - S^den + D (var0 + mu0^2)) / (gamma^num - gamma^den + D) - mu^2,
%    with one D per component, D = max(ebw_E * gamma^den, 2 * D_min), D_min the smallest
%    D keeping every dimension's variance and denominator positive.
%  * Discrete blocks (pi per channel, each transition row within a destination channel,
%    each coupling column via the source-selection counts, each state's mixture
%    weights): the weak-sense auxiliary update (lsim_disc_weak_sense), after adding
%    tau_d * p^ML to the numerator counts. Structural zeros stay zero.
%  * The domain projection (probability floor, mean box, variance bounds) is applied
%    last; info counts the entries it changed.
% No step-size search and no adaptive damping: this is the full update.
d = struct('ebw_E', 2, 'ismooth_tau', 100, 'ismooth_backoff', 'ml', 'ismooth_tau_discrete', 0, ...
    'weak_sense_iters', 100, 'weak_sense_tol', 1e-10, 'update_blocks', {{}});
f = fieldnames(d);
for j = 1:numel(f)
    if ~isfield(opts, f{j}), opts.(f{j}) = d.(f{j}); end
end
E = opts.ebw_E;
tau = opts.ismooth_tau;
tau_d = opts.ismooth_tau_discrete;
q = p;
sni = p.state_numbers_index;
D_used = [];
inner = [];
aux_ok = true;

for z = 1:p.C
    zi = sni(z)+1:sni(z+1);
    [q.pi_0(zi), it, ok] = discrete(p.pi_0(zi), num.pi(zi), den.pi(zi), dom.support_pi(zi), tau_d, opts);
    inner(end+1) = it; aux_ok = aux_ok && ok; %#ok<AGROW>
    for i = 1:sni(end)
        [row, it, ok] = discrete(p.transitions_matrices(i, zi)', num.A(i, zi)', den.A(i, zi)', ...
            dom.support_A(i, zi)', tau_d, opts);
        q.transitions_matrices(i, zi) = row';
        inner(end+1) = it; aux_ok = aux_ok && ok; %#ok<AGROW>
    end
    [q.coupling_tetha_IM(:, z), it, ok] = discrete(p.coupling_tetha_IM(:, z), num.theta(:, z), ...
        den.theta(:, z), dom.support_theta(:, z), tau_d, opts);
    inner(end+1) = it; aux_ok = aux_ok && ok; %#ok<AGROW>

    di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
    for i = 1:p.channel_num_states(z)
        K = p.num_gmm_component(z);
        [w, it, ok] = discrete(reshape(p.P_all(z, i, 1:K), K, 1), reshape(num.occ(z, i, 1:K), K, 1), ...
            reshape(den.occ(z, i, 1:K), K, 1), reshape(dom.support_w(z, i, 1:K), K, 1), tau_d, opts);
        q.P_all(z, i, 1:K) = w;
        inner(end+1) = it; aux_ok = aux_ok && ok; %#ok<AGROW>
        for k = 1:K
            [mu, v, Du] = gauss_update(p.mu_all(di, i, k), p.sigma_all(di, i, k), ...
                num.n(di, i, k), num.s(di, i, k), num.r(di, i, k), ...
                den.n(di, i, k), den.s(di, i, k), den.r(di, i, k), den.occ(z, i, k), E, tau, opts.ismooth_backoff);
            q.mu_all(di, i, k) = mu;
            q.sigma_all(di, i, k) = v;
            D_used(end+1) = Du; %#ok<AGROW>
        end
    end
end

if ~isempty(opts.update_blocks)
    fam = {'pi', 'pi_0'; 'A', 'transitions_matrices'; 'theta', 'coupling_tetha_IM'; ...
        'w', 'P_all'; 'mu', 'mu_all'; 'var', 'sigma_all'};
    bad = setdiff(opts.update_blocks, fam(:, 1));
    if ~isempty(bad)
        error('em_lsim_discriminative:badUpdateBlocks', 'Unknown update block(s): %s', strjoin(bad, ', '));
    end
    for j = 1:size(fam, 1)
        if ~ismember(fam{j, 1}, opts.update_blocks)
            q.(fam{j, 2}) = p.(fam{j, 2});
        end
    end
end

qp = lsim_disc_domain('project', q, dom);
fams = {'pi_0', 'transitions_matrices', 'coupling_tetha_IM', 'P_all', 'mu_all', 'sigma_all'};
info.projection_changed = struct();
total = 0;
for j = 1:numel(fams)
    nchg = nnz(qp.(fams{j}) ~= q.(fams{j}));
    info.projection_changed.(fams{j}) = nchg;
    total = total + nchg;
end
info.projection_changed_total = total;
q = qp;
info.max_D = max([D_used, 0]);
info.median_D = median(D_used);
info.weak_sense_max_iterations = max([inner, 0]);
info.weak_sense_monotone = aux_ok;
end

% ---------------------------------------------------------------------
function [pn, it, ok] = discrete(p0, gnum, gden, support, tau_d, opts)
pn = p0;
it = 0;
ok = true;
sup = support(:) & p0(:) > 0;
if nnz(sup) <= 1
    return
end
g = gnum(:);
g(~sup) = 0;
h = gden(:);
h(~sup) = 0;
if tau_d > 0 && sum(g) > 0
    g = g + tau_d*(g/sum(g));   % tau_d * p^ML pseudo-counts
end
pin = p0(:);
pin(~sup) = 0;
[x, inf_] = lsim_disc_weak_sense(g, h, pin, opts.weak_sense_iters, opts.weak_sense_tol);
pn(:) = x;
it = inf_.iterations;
ok = inf_.monotone;
end

function [mu_new, v_new, D] = gauss_update(mu, v, gn, xn, Sn, gd, xd, Sd, den_occ, E, tau, backoff)
% I-smoothing of the numerator, then the EBW update with D = max(E gamma^den, 2 D_min)
mu_new = mu;
v_new = v;
D = 0;
if tau > 0
    switch backoff
        case 'ml'
            mu_b = mu;
            v_b = v;
            ok = gn > 0;
            mu_b(ok) = xn(ok)./gn(ok);
            vb_ml = Sn(ok)./gn(ok) - mu_b(ok).^2;
            vb = v_b(ok);
            good = vb_ml > 0 & isfinite(vb_ml);
            vb(good) = vb_ml(good);
            v_b(ok) = vb;
            bad = ok;
            bad(ok) = ~good;
            mu_b(bad) = mu(bad);
        case 'previous'
            mu_b = mu;
            v_b = v;
        otherwise
            error('em_lsim_discriminative:badBackoff', 'ismooth_backoff must be ml or previous.');
    end
    gn = gn + tau;
    xn = xn + tau*mu_b;
    Sn = Sn + tau*(v_b + mu_b.^2);
end
n = gn - gd;
s = xn - xd;
r = Sn - Sd;
if all(n == 0) && all(s == 0) && den_occ == 0
    return
end
a = v;
b = r + n.*(v + mu.^2) - 2*s.*mu;
c = r.*n - s.^2;
disc = b.^2 - 4*a.*c;
root = -inf(size(mu));
has = disc >= 0;
root(has) = (-b(has) + sqrt(disc(has)))./(2*a(has));
D_min = max([0; root(:); -n(:)]);
D = max(E*den_occ, 2*D_min);
denom = n + D;
ok = denom > 1e-12*max(1, abs(D));
mu_new(ok) = (s(ok) + D*mu(ok))./denom(ok);
v_new(ok) = (r(ok) + D*(v(ok) + mu(ok).^2))./denom(ok) - mu_new(ok).^2;
bad = ~(v_new > 0) | ~isfinite(v_new) | ~isfinite(mu_new);
mu_new(bad) = mu(bad);
v_new(bad) = v(bad);
end
