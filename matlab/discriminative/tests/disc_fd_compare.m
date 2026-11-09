function res = disc_fd_compare(p, g, scorefun, h, families)
% Central finite differences of scorefun in the unconstrained coordinates
% (softmax logits per probability block, raw means, log variances), compared with
% the analytic gradient g (lsim_disc_gradient). Returns, per family, the max
% error relative to the largest finite-difference magnitude, and the cosine.
if nargin < 5 || isempty(families)
    families = {'pi', 'A', 'theta', 'w', 'mu', 'logv'};
end
fams = families;
res = struct('family', {}, 'n', {}, 'relerr', {}, 'cosine', {});
sni = p.state_numbers_index;
for f = 1:numel(fams)
    coords = {};
    switch fams{f}
        case 'pi'
            for z = 1:p.C
                blk = (sni(z)+1:sni(z+1))';
                for j = 1:numel(blk), coords{end+1} = {'pi_0', blk, j}; end %#ok<AGROW>
            end
        case 'A'
            sz = size(p.transitions_matrices);
            for z = 1:p.C
                for i = 1:sni(end)
                    blk = sub2ind(sz, i*ones(1, sni(z+1)-sni(z)), sni(z)+1:sni(z+1))';
                    for j = 1:numel(blk), coords{end+1} = {'transitions_matrices', blk, j}; end %#ok<AGROW>
                end
            end
        case 'theta'
            if p.C == 1, continue; end
            for z = 1:p.C
                blk = sub2ind([p.C p.C], (1:p.C), z*ones(1, p.C))';
                for j = 1:numel(blk), coords{end+1} = {'coupling_tetha_IM', blk, j}; end %#ok<AGROW>
            end
        case 'w'
            sz = size(p.P_all, 1:3);
            for z = 1:p.C
                K = p.num_gmm_component(z);
                if K < 2, continue; end
                for i = 1:p.channel_num_states(z)
                    blk = sub2ind(sz, z*ones(1, K), i*ones(1, K), 1:K)';
                    for j = 1:K, coords{end+1} = {'P_all', blk, j}; end %#ok<AGROW>
                end
            end
        case {'mu', 'logv'}
            fld = 'mu_all';
            if strcmp(fams{f}, 'logv'), fld = 'sigma_all'; end
            for z = 1:p.C
                di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
                for i = 1:p.channel_num_states(z)
                    for k = 1:p.num_gmm_component(z)
                        for d = di
                            coords{end+1} = {fld, sub2ind(size(p.(fld), 1:3), d, i, k), 0}; %#ok<AGROW>
                        end
                    end
                end
            end
    end
    if isempty(coords), continue; end
    an = zeros(numel(coords), 1);
    fd = zeros(numel(coords), 1);
    for c = 1:numel(coords)
        fld = coords{c}{1};
        blk = coords{c}{2};
        j = coords{c}{3};
        pp = perturb(p, fld, blk, j, h);
        pm = perturb(p, fld, blk, j, -h);
        fd(c) = (scorefun(pp) - scorefun(pm))/(2*h);
        switch fams{f}
            case 'pi', an(c) = g.pi(blk(j));
            case 'A', an(c) = g.A(blk(j));
            case 'theta', an(c) = g.theta(blk(j));
            case 'w', an(c) = g.w(blk(j));
            case 'mu', an(c) = g.mu(blk);
            case 'logv', an(c) = g.logv(blk);
        end
    end
    res(end+1) = struct('family', fams{f}, 'n', numel(coords), ...
        'relerr', max(abs(an - fd))/max(max(abs(fd)), 1e-8), ...
        'cosine', dot(an, fd)/(norm(an)*norm(fd) + eps)); %#ok<AGROW>
end
end

function q = perturb(p, fld, blk, j, h)
q = p;
if j == 0
    if strcmp(fld, 'sigma_all')
        q.sigma_all(blk) = exp(log(p.sigma_all(blk)) + h);   % log-variance coordinate
    else
        q.(fld)(blk) = p.(fld)(blk) + h;
    end
    return
end
u = log(p.(fld)(blk));
u(j) = u(j) + h;
e = exp(u - max(u));
q.(fld)(blk) = e/sum(e);
end

