function m = lsim_disc_metrics(truth, predicted, M)
% Balanced accuracy, macro-F1, plain accuracy and the confusion matrix
% (rows = true class, columns = predicted class) for labels in 1..M.
truth = truth(:);
predicted = predicted(:);
confusion = accumarray([truth, predicted], 1, [M, M]);
support = sum(confusion, 2);
predicted_count = sum(confusion, 1)';
tp = diag(confusion);
recall = tp ./ support;
precision = tp ./ predicted_count;
precision(predicted_count == 0) = 0;
f1 = 2*precision.*recall ./ (precision + recall);
f1(precision + recall == 0) = 0;
present = support > 0;
m.balanced_accuracy = mean(recall(present));
m.macro_f1 = mean(f1(present));
m.accuracy = sum(tp)/sum(support);
m.confusion = confusion;
m.recall = recall;
end
