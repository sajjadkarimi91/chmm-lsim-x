function thr = lsim_disc_tune_threshold(score, label, criterion)
%LSIM_DISC_TUNE_THRESHOLD Threshold maximizing balanced accuracy ('ba', default) or F1 ('f1') on a set.
%
%   thr = lsim_disc_tune_threshold(score, label, criterion)
%
% Tune on VALIDATION data and apply the threshold unchanged to test data (lsim_disc_threshold_metrics).
if nargin < 3 || isempty(criterion)
    criterion = 'ba';
end
m = lsim_disc_binary_metrics(score, label);
switch criterion
    case 'ba'
        thr = m.best_ba_threshold;
    case 'f1'
        thr = m.max_f1_threshold;
    otherwise
        error('lsim_disc_tune_threshold:badCriterion', 'criterion must be ''ba'' or ''f1''.');
end
end
