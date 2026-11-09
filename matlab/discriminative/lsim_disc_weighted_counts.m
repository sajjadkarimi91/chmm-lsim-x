function [num, den] = lsim_disc_weighted_counts(pk, data, w_num, w_den, fbc)
%LSIM_DISC_WEIGHTED_COUNTS Numerator / denominator block counts of one model.
%
%   [num, den] = lsim_disc_weighted_counts(pk, data, w_num, w_den, fbc)
%
% pk           : packed current model (lsim_disc_pack)
% data         : lsim_disc_prepare_data output (data.Y = sequences)
% w_num, w_den : N x 1 nonnegative per-sequence weights (a column of the objective's
%                parts.W.num / parts.W.den, lsim_disc_objective)
% fbc          : forward-backward constants (fbc.parallel = parfor over sequences)
%
% Every sequence's expected complete-data sufficient statistics are computed UNDER pk
% (lsim_disc_sequence_stats: the E-step quantities of em_lsim, including the coupling /
% transition source-selection counts of Karimi & Shamsollahi 2023), multiplied by the
% sequence's weight and summed. Sequences with zero weight on both sides are skipped (their
% statistics are never computed). Units: occupancies are weight x frames.
%
% Serial (fbc.parallel false): summed in sequence order.
% Parallel (fbc.parallel true): parfor over FIXED chunks of fbc.chunk_size weighted sequences
% (default 64); each worker accumulates its chunk's weighted counts locally and the chunk totals
% are added in chunk order. The result does not depend on the number of workers; it differs from
% the serial sum only by rounding (grouping).
if any(w_num < 0) || any(w_den < 0) || any(~isfinite([w_num(:); w_den(:)]))
    error('em_lsim_discriminative:badWeights', 'Statistic weights must be finite and nonnegative.');
end
num = lsim_disc_counts('zero', pk);
den = num;
idx = find(w_num(:) ~= 0 | w_den(:) ~= 0)';
if isempty(idx)
    return
end
if isfield(fbc, 'parallel') && fbc.parallel
    chunk = 64;
    if isfield(fbc, 'chunk_size') && ~isempty(fbc.chunk_size)
        chunk = fbc.chunk_size;
    end
    fbc_k = rmfield(fbc, intersect(fieldnames(fbc), {'parallel', 'chunk_size'}));
    starts = 1:chunk:numel(idx);
    nc = numel(starts);
    Yc = cell(1, nc); wn_c = cell(1, nc); wd_c = cell(1, nc);
    for c = 1:nc
        sel = idx(starts(c):min(starts(c) + chunk - 1, numel(idx)));
        Yc{c} = data.Y(sel);
        wn_c{c} = w_num(sel);
        wd_c{c} = w_den(sel);
    end
    parts_num = cell(1, nc);
    parts_den = cell(1, nc);
    parfor c = 1:nc
        [parts_num{c}, parts_den{c}] = accumulate(pk, Yc{c}, wn_c{c}, wd_c{c}, fbc_k);
    end
    for c = 1:nc
        num = lsim_disc_counts('plus', num, parts_num{c});
        den = lsim_disc_counts('plus', den, parts_den{c});
    end
    return
end
[num, den] = accumulate(pk, data.Y(idx), w_num(idx), w_den(idx), fbc);
end

function [num, den] = accumulate(pk, Y, wn, wd, fbc)
% weighted accumulation in sequence order
num = lsim_disc_counts('zero', pk);
den = num;
for n = 1:numel(Y)
    st = lsim_disc_sequence_stats(pk, Y{n}, fbc);
    if wn(n) > 0
        num = lsim_disc_counts('add', num, st, wn(n), pk);
    end
    if wd(n) > 0
        den = lsim_disc_counts('add', den, st, wd(n), pk);
    end
end
end
