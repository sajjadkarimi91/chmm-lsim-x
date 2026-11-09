function [p, info] = lsim_disc_weak_sense(gnum, gden, p0, max_iter, tol)
%LSIM_DISC_WEAK_SENSE Weak-sense auxiliary update of one probability vector (published MMI).
%
%   [p, info] = lsim_disc_weak_sense(gnum, gden, p0, max_iter, tol)
%
% Maximizes, over the probability simplex on the support of p0 (p0 > 0),
%     G(p) = sum_j [ gnum_j log p_j - gden_j p_j / p0_j ]
% by the iteration (weak-sense auxiliary function; Woodland & Povey 2002; Povey 2003)
%     p_j <- (gnum_j + k_j p_j) / sum_i (gnum_i + k_i p_i),
%     k_j  = max_i(gden_i / p0_i) - gden_j / p0_j   (>= 0),
% started at p0, for at most max_iter iterations or until the largest relative
% change is below tol. It is a minorize-maximize step (k_j p_j >= k_j p_j' (1 + log p_j
% - log p_j') for k_j >= 0), so G never decreases. Entries outside the support stay
% exactly zero. With gden = 0 one iteration gives the EM estimate gnum / sum(gnum).
%
% info.iterations, info.aux (G after each iteration, starting with G(p0)),
% info.monotone (true if G never decreased beyond 1e-12 relative).
gnum = gnum(:);
gden = gden(:);
shape = size(p0);
p0 = p0(:);
p = p0;
sup = p0 > 0;
info = struct('iterations', 0, 'aux', [], 'monotone', true);
if nnz(sup) <= 1 || (sum(gnum(sup)) <= 0 && all(gden(sup) == 0))
    p = reshape(p, shape);
    return
end
if any(gnum(sup) < 0) || any(gden(sup) < 0)
    error('lsim_disc_weak_sense:negativeCounts', 'Numerator and denominator counts must be nonnegative.');
end
g = gnum(sup);
c = gden(sup)./p0(sup);
k = max(c) - c;
x = p0(sup);
aux = @(y) sum(g(g > 0).*log(y(g > 0))) - sum(c.*y);
G = zeros(max_iter + 1, 1);
G(1) = aux(x);
it = 0;
for it = 1:max_iter
    y = g + k.*x;
    y = y/sum(y);
    G(it + 1) = aux(y);
    change = max(abs(y - x)./max(x, realmin));
    x = y;
    if change < tol
        break
    end
end
G = G(1:it + 1);
p(sup) = x;
p(~sup) = 0;
p = reshape(p, shape);
info.iterations = it;
info.aux = G;
info.monotone = all(diff(G) >= -1e-12*max(1, abs(G(1:end-1))));
end
