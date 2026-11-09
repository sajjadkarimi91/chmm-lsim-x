function t = lsim_disc_threshold_metrics(score, label, threshold)
%LSIM_DISC_THRESHOLD_METRICS Balanced accuracy, F1, sensitivity, specificity, precision at a threshold.
%
%   t = lsim_disc_threshold_metrics(score, label, threshold)
%
% Positive when score >= threshold (NaN scores are negative); label: logical, true = positive.
s = double(score(:));
s(isnan(s)) = -Inf;
y = logical(label(:));
pred = s >= threshold;
tp = nnz(pred & y); fp = nnz(pred & ~y); fn = nnz(~pred & y); tn = nnz(~pred & ~y);
sens = NaN; spec = NaN;
if tp + fn > 0, sens = tp/(tp + fn); end
if tn + fp > 0, spec = tn/(tn + fp); end
prec = 0;
if tp + fp > 0, prec = tp/(tp + fp); end
f1 = 0;
if 2*tp + fp + fn > 0, f1 = 2*tp/(2*tp + fp + fn); end
t = struct('threshold', threshold, 'balanced_accuracy', 0.5*(sens + spec), 'f1', f1, 'sensitivity', sens, ...
    'specificity', spec, 'precision', prec, 'tp', tp, 'fp', fp, 'fn', fn, 'tn', tn);
end
