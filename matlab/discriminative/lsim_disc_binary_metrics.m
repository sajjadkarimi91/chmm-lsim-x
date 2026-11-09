function m = lsim_disc_binary_metrics(score, label)
%LSIM_DISC_BINARY_METRICS AUROC, AUPRC, max-F1 and best-threshold balanced accuracy of a detection score.
%
%   m = lsim_disc_binary_metrics(score, label)
%
% score : N x 1, higher = more positive (e.g. the LLR of lsim_disc_detection_score); NaN ranks lowest
% label : N x 1 logical (true = positive class)
% m     : .auroc (trapezoidal ROC area = Mann-Whitney U with average ranks for ties),
%         .auprc (average precision, sum_k (R_k - R_{k-1}) P_k, as scikit-learn),
%         .max_f1 / .max_f1_threshold, .ba_best_threshold / .best_ba_threshold (best balanced accuracy
%         over every threshold, Inf = everything negative), .n_pos, .n_neg.
% Threshold convention: positive when score >= threshold. Python twin: chmm_lsim.discriminative.metrics.
s = double(score(:));
y = logical(label(:));
s(isnan(s)) = -Inf;
s = min(max(s, -realmax), realmax);
P = nnz(y);
N = nnz(~y);
m = struct('auroc', NaN, 'auprc', NaN, 'max_f1', NaN, 'max_f1_threshold', NaN, ...
    'ba_best_threshold', NaN, 'best_ba_threshold', NaN, 'n_pos', P, 'n_neg', N);
if P == 0 || N == 0
    return
end
[ss, order] = sort(s, 'descend');
ys = double(y(order));
last = [find(diff(ss) ~= 0); numel(ss)];      % last index of every tie block
ctp = cumsum(ys);
tp = ctp(last);
fp = last - tp;
thr = ss(last);
tpr = [0; tp/P];
fpr = [0; fp/N];
m.auroc = sum((fpr(2:end) - fpr(1:end-1)).*(tpr(2:end) + tpr(1:end-1))/2);
precision = tp./(tp + fp);
recall = tp/P;
m.auprc = sum(diff([0; recall]).*precision);
f1 = 2*tp./(2*tp + fp + (P - tp));
[m.max_f1, j] = max(f1);
m.max_f1_threshold = thr(j);
ba = 0.5*(tp/P + (N - fp)/N);
[b, k] = max(ba);
if b >= 0.5
    m.ba_best_threshold = b;
    m.best_ba_threshold = thr(k);
else
    m.ba_best_threshold = 0.5;
    m.best_ba_threshold = Inf;
end
end
