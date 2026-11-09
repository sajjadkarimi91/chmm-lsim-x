function out = lsim_disc_counts(op, varargin)
% Block-level expected counts, the currency every update and gradient uses.
%
%   bc = lsim_disc_counts('zero', p)            all-zero counts for model p
%   bc = lsim_disc_counts('add', bc, st, w, p)   bc += w * block counts of stats st
%   bc = lsim_disc_counts('plus', bc1, bc2)      elementwise sum
%
% Fields
%   pi    S x 1            initial-state counts
%   A     S x S            transition counts: row block (i, z) is a probability
%                          block over destination states j in z (= xi)
%   theta C x C            coupling counts: column z is a probability block over
%                          source channels c, theta(c,z) = sum_{i in c, j in z} xi(i,j)
%   occ   C x maxS x maxK  mixture-weight counts per (channel, state) block
%   n,s,r D x maxS x maxK  Gaussian occupancy, first and second moments (masked)
%
% A and theta are kept separately (rather than deriving theta from A later) so
% that each block's update sees its own counts.

switch op
    case 'zero'
        p = varargin{1};
        S = p.state_numbers_index(end);
        maxS = max(p.channel_num_states);
        maxK = max(p.num_gmm_component);
        D = p.dimension_numbers_index(end);
        out = struct('pi', zeros(S, 1), 'A', zeros(S, S), 'theta', zeros(p.C, p.C), ...
            'occ', zeros(p.C, maxS, maxK), 'n', zeros(D, maxS, maxK), ...
            's', zeros(D, maxS, maxK), 'r', zeros(D, maxS, maxK));

    case 'add'
        [bc, st, w, p] = varargin{:};
        bc.pi = bc.pi + w*st.pi;
        bc.A = bc.A + w*st.xi;
        bc.theta = bc.theta + w*block_sum_xi(st.xi, p);
        bc.occ = bc.occ + w*st.occ;
        bc.n = bc.n + w*st.n;
        bc.s = bc.s + w*st.s;
        bc.r = bc.r + w*st.r;
        out = bc;

    case 'plus'
        [a, b] = varargin{:};
        out = a;
        f = fieldnames(a);
        for j = 1:numel(f)
            out.(f{j}) = a.(f{j}) + b.(f{j});
        end

    otherwise
        error('lsim_disc_counts: unknown op %s', op);
end

end

function th = block_sum_xi(xi, p)
th = zeros(p.C, p.C);
sni = p.state_numbers_index;
for c = 1:p.C
    for z = 1:p.C
        th(c, z) = sum(sum(xi(sni(c)+1:sni(c+1), sni(z)+1:sni(z+1))));
    end
end
end
