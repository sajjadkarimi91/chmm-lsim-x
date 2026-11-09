classdef TestValidationSelection < matlab.unittest.TestCase
    % Validation metrics, checkpoint selection and early stopping of em_lsim_discriminative, and the
    % metric functions lsim_disc_binary_metrics / _score_metrics / _tune_threshold /
    % _threshold_metrics / _evaluate (Python twin: chmm_lsim.discriminative.metrics), for both MMI
    % recipes.
    %
    %   addpath(genpath('matlab')); results = runtests('TestValidationSelection')

    properties
        tr
        va
        init
    end

    properties (TestParameter)
        recipe = {'published', 'legacy'}
    end

    methods (TestClassSetup)
        function setup(tc)
            [tc.tr, tc.init] = disc_test_problems('lsim', 31, 30);
            [tc.va, ~] = disc_test_problems('lsim', 32, 30);
        end
    end

    methods (Test)
        function binaryMetricsMatchBruteForce(tc)
            rng(5, 'twister');
            for trial = 1:6
                n = 300;
                y = rand(n, 1) < 0.2 + 0.1*trial;
                s = randn(n, 1) + 1.2*y;
                if mod(trial, 2), s = round(s, 1); end          % ties
                if trial == 6, s(1:3) = NaN; end                % unscorable sequences rank lowest
                m = lsim_disc_binary_metrics(s, y);
                sn = s; sn(isnan(sn)) = -Inf;
                P = nnz(y); N = nnz(~y);
                % AUROC: Mann-Whitney with ties counted 1/2
                [a, b] = meshgrid(sn(y), sn(~y));
                tc.verifyEqual(m.auroc, (nnz(a > b) + 0.5*nnz(a == b))/(P*N), 'AbsTol', 1e-12);
                % AUPRC (average precision), max-F1, best BA: every distinct threshold
                thr = unique(sn);
                ap = 0; prev_r = 0; f1 = []; ba = 0.5;
                for t = flip(thr(:))'
                    pred = sn >= t;
                    tp = nnz(pred & y); fp = nnz(pred & ~y);
                    r = tp/P; pr = tp/(tp + fp);
                    ap = ap + (r - prev_r)*pr; prev_r = r;
                    f1(end+1) = 2*tp/(2*tp + fp + P - tp); %#ok<AGROW>
                    ba = max(ba, 0.5*(tp/P + (N - fp)/N));
                end
                tc.verifyEqual(m.auprc, ap, 'AbsTol', 1e-12);
                tc.verifyEqual(m.max_f1, max(f1), 'AbsTol', 1e-12);
                tc.verifyEqual(m.ba_best_threshold, ba, 'AbsTol', 1e-12);
                t = lsim_disc_threshold_metrics(s, y, lsim_disc_tune_threshold(s, y, 'f1'));
                tc.verifyEqual(t.f1, m.max_f1, 'AbsTol', 1e-12);
                t = lsim_disc_threshold_metrics(s, y, lsim_disc_tune_threshold(s, y, 'ba'));
                tc.verifyEqual(t.balanced_accuracy, m.ba_best_threshold, 'AbsTol', 1e-12);
            end
        end

        function defaultSelectionUnchanged(tc, recipe)
            ex = struct('verbose', 0, 'validation_observations', {tc.va}, 'tol_rel', 0, 'mmi_recipe', recipe);
            [~, h, d] = em_lsim_discriminative(tc.tr, [2 2], [1 1], 6, ex, tc.init);
            ex.selection_metric = 'balanced_accuracy';
            [~, h2, d2] = em_lsim_discriminative(tc.tr, [2 2], [1 1], 6, ex, tc.init);
            tc.verifyEqual(h2.objective, h.objective);
            tc.verifyEqual(d2.checkpoint_iteration, d.checkpoint_iteration);
            v = h.val_balanced_accuracy;
            best = find(v == max(v));
            [~, j] = max(h.val_objective(best));
            tc.verifyEqual(d.checkpoint_iteration, best(j) - 1);
            tc.verifyEqual(h.val_selection_value, h.val_balanced_accuracy);
            tc.verifyTrue(all(isfinite([h.val_auroc; h.val_auprc; h.val_max_f1; h.val_ba_best_threshold; ...
                h.train_auroc; h.train_auprc; h.train_max_f1])));
        end

        function selectionMetricPicksBestIteration(tc, recipe)
            metrics = {'auroc', 'auprc', 'max_f1', 'ba_best_threshold', 'objective'};
            fields = {'val_auroc', 'val_auprc', 'val_max_f1', 'val_ba_best_threshold', 'val_objective'};
            for j = 1:numel(metrics)
                ex = struct('verbose', 0, 'validation_observations', {tc.va}, 'tol_rel', 0, ...
                    'selection_metric', metrics{j}, 'mmi_recipe', recipe);
                [models, h, d] = em_lsim_discriminative(tc.tr, [2 2], [1 1], 6, ex, tc.init);
                v = h.(fields{j});
                best = find(v == max(v));
                [~, k] = max(h.val_objective(best));
                tc.verifyEqual(d.checkpoint_iteration, best(k) - 1, metrics{j});
                tc.verifyEqual(d.checkpoint_validation_value, max(v), metrics{j});
                tc.verifyEqual(models{1}.meta.iteration, d.checkpoint_iteration);
            end
        end

        function earlyStopping(tc, recipe)
            ex = struct('verbose', 0, 'validation_observations', {tc.va}, 'tol_rel', 0, ...
                'selection_metric', 'auroc', 'early_stopping_patience', 2, 'mmi_recipe', recipe);
            [~, h, d] = em_lsim_discriminative(tc.tr, [2 2], [1 1], 40, ex, tc.init);
            n = d.iterations_run;
            if startsWith(d.stop_reason, 'early stopping')
                v = h.val_auroc;
                tc.verifyLessThanOrEqual(max(v(end-1:end)), max(v(1:end-2)));
                tc.verifyEqual(v(d.checkpoint_iteration + 1), max(v));
                tc.verifyLessThan(n, 40);
            end
            tc.verifyEqual(numel(h.objective), n + 1);
        end

        function errors(tc)
            ex = struct('verbose', 0, 'selection_metric', 'auroc');
            tc.verifyError(@() em_lsim_discriminative(tc.tr, [2 2], [1 1], 2, ex, tc.init), ...
                'em_lsim_discriminative:needValidation');
            ex = struct('verbose', 0, 'early_stopping_patience', 3);
            tc.verifyError(@() em_lsim_discriminative(tc.tr, [2 2], [1 1], 2, ex, tc.init), ...
                'em_lsim_discriminative:needValidation');
            tc.verifyError(@() lsim_disc_options(struct('selection_metric', 'accuracy')), ...
                'em_lsim_discriminative:badOptionValue');
        end

        function evaluateTunesOnValidation(tc)
            ev = lsim_disc_evaluate(tc.init, struct('train', {tc.tr}, 'val', {tc.va}));
            tc.verifyNotEmpty(ev.threshold);
            tc.verifyEqual(ev.val.balanced_accuracy_at_tuned, ev.val.ba_best_threshold, 'AbsTol', 1e-12);
            for f = {'auroc', 'auprc', 'max_f1', 'balanced_accuracy', 'f1_at_tuned'}
                tc.verifyTrue(isfinite(ev.train.(f{1})));
            end
            tc.verifySize(ev.scores.val, [numel(ev.labels.val), 2]);
        end
    end
end
