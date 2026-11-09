function out = lsim_disc_domain(op, varargin)
% The compact feasible domain the discriminative trainer optimizes over.
%
% Discriminative objectives reward lowering competing classes' likelihoods,
% which can be driven to zero by moving a mean away, or by starving a transition,
% without bound; variance floors alone do not prevent that. Training is therefore
% restricted to a compact, convex set derived from TRAINING data only:
%
%   probabilities   every entry of every probability block that is on the
%                   block's support is >= prob_floor; entries that are exactly
%                   zero in the initial model are STRUCTURAL zeros and stay zero
%   means           mu(d) in [lo(d), hi(d)], lo/hi = pooled training min/max
%                   widened by mean_margin*(max - min) on each side
%   variances       v(d) in [var_lo_frac, var_hi_frac] * pooled training var(d)
%
% All three constraints are convex, so the segment between two feasible models
% stays feasible -- which is what lets the trainer backtrack along the segment
% to an EBW proposal without ever leaving the domain.
%
%   dom = lsim_disc_domain('build', data, p_init, opts)
%   p   = lsim_disc_domain('project', p, dom)     move a model into the domain
%   ok  = lsim_disc_domain('check', p, dom, tol)  feasibility check (tests)
%   b   = lsim_disc_domain('at_bounds', p, dom)   fractions of parameters at the bounds:
%         .prob_at_floor (supported entries of pi, A, theta, mixture weights within 1e-9 of
%         the floor, blocks with >= 2 supported entries), .var_at_lower, .var_at_cap,
%         .mean_at_bound (fractions of the used variance / mean entries)

switch op
    case 'build'
        [data, p, opts] = varargin{:};
        D = p.dimension_numbers_index(end);
        allY = cat(2, data.Y{:});
        lo = min(allY, [], 2, 'omitnan');
        hi = max(allY, [], 2, 'omitnan');
        v = var(allY, [], 2, 'omitnan');
        if any(~isfinite([lo; hi; v])) || any(v <= 0)
            error('em_lsim_discriminative:degenerateFeature', ...
                'Every feature dimension needs finite, non-constant training data.');
        end
        span = hi - lo;
        out.mu_lo = lo - opts.mean_margin*span;
        out.mu_hi = hi + opts.mean_margin*span;
        out.var_lo = opts.var_lo_frac*v;
        out.var_hi = opts.var_hi_frac*v;
        out.prob_floor = opts.prob_floor;
        out.D = D;
        % structural-zero masks, from the initial model
        out.support_pi = p.pi_0 > 0;
        out.support_A = p.transitions_matrices > 0;
        out.support_theta = p.coupling_tetha_IM > 0;
        out.support_w = p.P_all > 0;

    case 'project'
        [p, dom] = varargin{:};
        sni = p.state_numbers_index;
        for z = 1:p.C
            zi = sni(z)+1:sni(z+1);
            p.pi_0(zi) = floor_block(p.pi_0(zi), dom.support_pi(zi), dom.prob_floor);
            for c = 1:p.C
                ci = sni(c)+1:sni(c+1);
                for i = ci
                    p.transitions_matrices(i, zi) = floor_block(p.transitions_matrices(i, zi), ...
                        dom.support_A(i, zi), dom.prob_floor);
                end
            end
            p.coupling_tetha_IM(:, z) = floor_block(p.coupling_tetha_IM(:, z), ...
                dom.support_theta(:, z), dom.prob_floor);
            for i = 1:p.channel_num_states(z)
                K = p.num_gmm_component(z);
                w = reshape(p.P_all(z, i, 1:K), K, 1);
                sw = reshape(dom.support_w(z, i, 1:K), K, 1);
                p.P_all(z, i, 1:K) = floor_block(w, sw, dom.prob_floor);
                di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
                for k = 1:K
                    p.mu_all(di, i, k) = min(max(p.mu_all(di, i, k), dom.mu_lo(di)), dom.mu_hi(di));
                    p.sigma_all(di, i, k) = min(max(p.sigma_all(di, i, k), dom.var_lo(di)), dom.var_hi(di));
                end
            end
        end
        out = p;

    case 'check'
        [p, dom, tol] = varargin{:};
        ok = true;
        sni = p.state_numbers_index;
        for z = 1:p.C
            zi = sni(z)+1:sni(z+1);
            ok = ok && block_ok(p.pi_0(zi), dom.support_pi(zi), dom.prob_floor, tol);
            for i = 1:sni(end)
                ok = ok && block_ok(p.transitions_matrices(i, zi), dom.support_A(i, zi), dom.prob_floor, tol);
            end
            ok = ok && block_ok(p.coupling_tetha_IM(:, z), dom.support_theta(:, z), dom.prob_floor, tol);
            for i = 1:p.channel_num_states(z)
                K = p.num_gmm_component(z);
                ok = ok && block_ok(reshape(p.P_all(z, i, 1:K), K, 1), ...
                    reshape(dom.support_w(z, i, 1:K), K, 1), dom.prob_floor, tol);
                di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
                for k = 1:K
                    mu = p.mu_all(di, i, k);
                    v = p.sigma_all(di, i, k);
                    ok = ok && all(mu >= dom.mu_lo(di)-tol) && all(mu <= dom.mu_hi(di)+tol) ...
                        && all(v >= dom.var_lo(di)*(1-tol)) && all(v <= dom.var_hi(di)*(1+tol));
                end
            end
        end
        out = ok;

    case 'at_bounds'
        [p, dom] = varargin{:};
        sni = p.state_numbers_index;
        rel = 1e-9;
        nprob = 0; nfloor = 0; nvar = 0; nlo = 0; nhi = 0; nmean = 0; nmb = 0;
        blocks = {};
        for z = 1:p.C
            zi = sni(z)+1:sni(z+1);
            blocks{end+1} = {p.pi_0(zi), dom.support_pi(zi)}; %#ok<AGROW>
            for i = 1:sni(end)
                blocks{end+1} = {p.transitions_matrices(i, zi), dom.support_A(i, zi)}; %#ok<AGROW>
            end
            blocks{end+1} = {p.coupling_tetha_IM(:, z), dom.support_theta(:, z)}; %#ok<AGROW>
            di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
            for i = 1:p.channel_num_states(z)
                K = p.num_gmm_component(z);
                blocks{end+1} = {reshape(p.P_all(z, i, 1:K), K, 1), reshape(dom.support_w(z, i, 1:K), K, 1)}; %#ok<AGROW>
                for k = 1:K
                    v = p.sigma_all(di, i, k);
                    mu = p.mu_all(di, i, k);
                    nvar = nvar + numel(v);
                    nlo = nlo + nnz(v <= dom.var_lo(di)*(1 + rel));
                    nhi = nhi + nnz(v >= dom.var_hi(di)*(1 - rel));
                    nmean = nmean + numel(mu);
                    nmb = nmb + nnz(mu <= dom.mu_lo(di) + rel*abs(dom.mu_lo(di)) | mu >= dom.mu_hi(di) - rel*abs(dom.mu_hi(di)));
                end
            end
        end
        for j = 1:numel(blocks)
            q = blocks{j}{1}(:);
            sup = blocks{j}{2}(:);
            if nnz(sup) >= 2
                nprob = nprob + nnz(sup);
                nfloor = nfloor + nnz(q(sup) <= dom.prob_floor*(1 + rel));
            end
        end
        out = struct('prob_at_floor', nfloor/max(nprob, 1), 'var_at_lower', nlo/max(nvar, 1), ...
            'var_at_cap', nhi/max(nvar, 1), 'mean_at_bound', nmb/max(nmean, 1));

    otherwise
        error('lsim_disc_domain: unknown op %s', op);
end

end

function q = floor_block(q, support, floor_value)
% Returns the block restricted to its support, renormalized, with every
% supported entry >= floor_value; unsupported (structural) entries are exactly
% 0. If some entry is below the floor, the block is mixed with the uniform
% distribution on its support by the smallest weight that lifts the minimum to
% the floor -- a convex operation, so the result is still a distribution.
shape = size(q);
q = q(:);
support = support(:);
q(~support) = 0;
m = nnz(support);
if m == 0
    q = reshape(q, shape);
    return
end
if floor_value*m >= 1
    error('em_lsim_discriminative:floorTooLarge', ...
        'prob_floor %g is infeasible for a block with %d supported entries.', floor_value, m);
end
q(support) = q(support)/sum(q(support));
low = min(q(support));
if low < floor_value
    u = 1/m;
    lambda = (floor_value - low)/(u - low);
    q(support) = (1-lambda)*q(support) + lambda*u;
end
q = reshape(q, shape);
end

function ok = block_ok(q, support, floor_value, tol)
q = q(:);
support = support(:);
ok = all(q(~support) == 0) && abs(sum(q) - 1) <= tol && ...
    (~any(support) || min(q(support)) >= floor_value*(1-tol));
end
