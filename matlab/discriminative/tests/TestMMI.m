classdef TestMMI < matlab.unittest.TestCase
    % Unit tests of MMI (em_lsim_discriminative with objective = 'mmi'):
    %   J = sum_n a_n log q(n,y_n), q = softmax(κ L + log pi), a_n = 1/(M N_y);
    % the published recipe (Woodland & Povey 2002; Povey et al. 2008: full EBW update with
    % D = max(E gamma_den, 2 D_min), I-smoothing, weak-sense discrete updates) and the legacy
    % recipe (backtracking, adaptive damping, no I-smoothing).
    %
    %   addpath(genpath('matlab')); results = runtests('TestMMI')

    methods (TestClassSetup)
        function addPaths(~)
            here = fileparts(mfilename('fullpath'));
            addpath(genpath(fileparts(fileparts(here))));
        end
    end

    methods (Test)
        function objectiveAndScoreGradient(tc)
            % J matches its formula; dJ/dL = a_n κ (1[k=y] - q) matches central finite
            % differences; legacy weights W.num - W.den equal that gradient exactly; published
            % weights equal N/κ times it (frame units, κ only inside q)
            rng(110, 'twister');
            for M = 2:4
                [L, label, T] = disc_test_problems('scores', 9, M);
                lp = log(rand(M, 1) + 0.2); lp = lp - log(sum(exp(lp)));
                for kap = [1 0.1 1e-3]
                    cl = struct('objective', 'mmi', 'mmi_recipe', 'legacy', 'score_scale', kap, 'log_prior', lp);
                    [J, pm] = lsim_disc_objective(L, label, T, cl);
                    Z = kap*L + lp';
                    Np = accumarray(label, 1, [M, 1]);
                    own = sub2ind(size(L), (1:numel(label))', label);
                    ref = sum(1./(M*Np(label)).*(Z(own) - log(sum(exp(Z), 2))));
                    tc.verifyLessThan(abs(J - ref), 1e-12*max(1, abs(ref)));
                    g = -pm.a.*kap.*pm.q;
                    g(own) = g(own) + pm.a*kap;
                    tc.verifyLessThan(max(abs(pm.W.num(:) - pm.W.den(:) - g(:))), 1e-14);
                    fd = zeros(size(L));
                    for i = 1:numel(L)
                        Lp = L; Lp(i) = Lp(i) + 1e-5;
                        Lm = L; Lm(i) = Lm(i) - 1e-5;
                        fd(i) = (lsim_disc_objective(Lp, label, T, cl) - lsim_disc_objective(Lm, label, T, cl))/2e-5;
                    end
                    tc.verifyLessThan(max(abs(fd(:) - g(:)))/max(abs(g(:))), 1e-6, sprintf('M=%d kappa=%g', M, kap));
                    cp = cl; cp.mmi_recipe = 'published'; cp.N_stat = numel(label);
                    [Jp, pp] = lsim_disc_objective(L, label, T, cp);
                    tc.verifyEqual(Jp, J);
                    tc.verifyLessThan(max(abs((pp.W.num(:) - pp.W.den(:)) - numel(label)/kap*g(:))), 1e-10*numel(label)/kap*max(abs(g(:))));
                end
            end
        end

        function legacyAcceptedObjectiveIsMonotone(tc)
            [tr, init] = disc_test_problems('lsim', 107);
            for kap = [1 0.1]
                ex = struct('objective', 'mmi', 'mmi_recipe', 'legacy', 'score_scale', kap, 'verbose', 0);
                [~, h, dg] = em_lsim_discriminative(tr, [2 2], [1 1], 10, ex, init);
                tc.verifyTrue(all(diff(h.objective) >= 0));
                tc.verifyEqual(dg.returned_objective_recomputed, dg.returned_objective_in_history);
                tc.verifyTrue(all(isfinite(h.frac_q_uncertain)));
            end
        end

        function publishedTrainerRuns(tc)
            [tr, init] = disc_test_problems('lsim', 402);
            ex = struct('objective', 'mmi', 'score_scale', 0.1, 'ismooth_tau', 10, 'ismooth_tau_discrete', 1, 'verbose', 0);
            [~, h, dg] = em_lsim_discriminative(tr, [2 2], [1 1], [], ex, init);
            tc.verifyEqual(numel(h.objective), 9, 'published default: 8 iterations');
            tc.verifyTrue(all(isfinite(h.objective)));
            tc.verifyTrue(all(h.step_tau(2:end, :) == 1, 'all'));
            tc.verifyTrue(all(isfinite(h.D_max(2:end, :)), 'all') && all(h.weak_sense_monotone(:)));
            tc.verifyTrue(all(isnan(h.damping_E(:))));
            tc.verifyTrue(all(isfinite(h.projection_changed(2:end, :)), 'all'));
            tc.verifyEqual(dg.stop_reason, 'max_iteration');
        end

        function zeroDenominatorIsEmStep(tc)
            % zero denominator statistics, tau > 0 (ML back-off): the published update equals
            % the EM M-step of the numerator statistics
            [p, num, ~, dom] = disc_test_problems('counts', 201);
            den = lsim_disc_counts('zero', p);
            q = lsim_disc_published_update(p, num, den, dom, struct('ismooth_tau', 37, 'ebw_E', 2));
            tc.verifyLessThan(disc_test_problems('max_block_diff', q, disc_test_problems('em_from_counts', p, num)), 1e-10);
        end

        function gaussianUpdateMatchesFormula(tc)
            % tau = 0: mu, var = EBW formula with D = max(E gamma_den, 2 D_min), D_min by bisection
            [p, num, den, dom] = disc_test_problems('counts', 205);
            E = 2;
            q = lsim_disc_published_update(p, num, den, dom, struct('ismooth_tau', 0, 'ebw_E', E));
            worst = 0;
            for z = 1:p.C
                di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
                for i = 1:p.channel_num_states(z)
                    mu = p.mu_all(di, i, 1); v = p.sigma_all(di, i, 1);
                    n = num.n(di, i, 1) - den.n(di, i, 1);
                    s = num.s(di, i, 1) - den.s(di, i, 1);
                    r = num.r(di, i, 1) - den.r(di, i, 1);
                    D = max(E*den.occ(z, i, 1), 2*bisect_dmin(mu, v, n, s, r));
                    mun = (s + D*mu)./(n + D);
                    vn = (r + D*(v + mu.^2))./(n + D) - mun.^2;
                    worst = max([worst; abs(q.mu_all(di, i, 1) - mun)./max(1, abs(mun)); abs(q.sigma_all(di, i, 1) - vn)./vn]);
                end
            end
            tc.verifyLessThan(worst, 1e-7);
        end

        function dRuleGivesPositiveVariances(tc)
            for seed = 206:215
                [p, num, den, dom] = disc_test_problems('counts', seed, 5);
                q = lsim_disc_published_update(p, num, den, dom, struct('ismooth_tau', 0));
                tc.verifyTrue(all(q.sigma_all(:) > 0) && all(isfinite(q.mu_all(:))));
            end
        end

        function weakSenseAuxiliaryNonDecreasing(tc)
            rng(301, 'twister');
            for t = 1:200
                m = randi([2 7]);
                p0 = rand(m, 1) + 0.01; p0 = p0/sum(p0);
                [~, info] = lsim_disc_weak_sense(rand(m, 1)*10.*(rand(m, 1) > 0.2), rand(m, 1)*10, p0, 100, 0);
                tc.verifyTrue(info.monotone);
            end
        end

        function weakSenseMatchesFmincon(tc)
            tc.assumeTrue(license('test', 'Optimization_Toolbox') && exist('fmincon', 'file') == 2, 'fmincon not available');
            rng(302, 'twister');
            o = optimoptions('fmincon', 'Display', 'off', 'OptimalityTolerance', 1e-14, 'StepTolerance', 1e-14, ...
                'ConstraintTolerance', 1e-14, 'MaxIterations', 2000, 'Algorithm', 'interior-point');
            for t = 1:20
                m = randi([2 6]);
                p0 = rand(m, 1) + 0.05; p0 = p0/sum(p0);
                g = rand(m, 1)*10 + 0.1;
                h = rand(m, 1)*10;
                p = lsim_disc_weak_sense(g, h, p0, 20000, 1e-15);
                x = fmincon(@(x) -(sum(g.*log(x)) - sum(h./p0.*x)), p0, [], [], ones(1, m), 1, 1e-12*ones(m, 1), ones(m, 1), [], o);
                tc.verifyLessThan(max(abs(p - x)), 1e-6, sprintf('case %d', t));
            end
        end

        function weakSenseKeepsStructuralZeros(tc)
            p0 = [0.3; 0; 0.5; 0; 0.2];
            [p, info] = lsim_disc_weak_sense([3; 9; 1; 4; 2], [1; 5; 2; 0; 1], p0, 100, 1e-12);
            tc.verifyEqual(p([2 4]), [0; 0]);
            tc.verifyLessThan(abs(sum(p) - 1), 1e-14);
            tc.verifyTrue(info.monotone);
            [pm, ~, ~, dom] = disc_test_problems('counts', 303);
            pm.transitions_matrices(1, 2) = 0;
            dom.support_A(1, 2) = false;
            pm = lsim_disc_domain('project', pm, dom);
            [~, num, den] = disc_test_problems('counts', 303);
            q = lsim_disc_published_update(pm, num, den, dom, struct());
            tc.verifyEqual(q.transitions_matrices(1, 2), 0);
        end

        function iSmoothingLimitIsBackoff(tc)
            [p, num, den, dom] = disc_test_problems('counts', 401);
            q = lsim_disc_published_update(p, num, den, dom, struct('ismooth_tau', 1e12, 'ismooth_backoff', 'ml'));
            ref = disc_test_problems('em_from_counts', p, num);
            tc.verifyLessThan(max(abs(q.mu_all(:) - ref.mu_all(:))), 1e-6);
            tc.verifyLessThan(max(abs(q.sigma_all(:) - ref.sigma_all(:))./ref.sigma_all(:)), 1e-6);
            q = lsim_disc_published_update(p, num, den, dom, struct('ismooth_tau', 1e12, 'ismooth_backoff', 'previous'));
            tc.verifyLessThan(max(abs(q.mu_all(:) - p.mu_all(:))), 1e-6);
            tc.verifyLessThan(max(abs(q.sigma_all(:) - p.sigma_all(:))./p.sigma_all(:)), 1e-6);
        end

        function updateDirectionMatchesFirstOrder(tc)
            out = update_direction_checks(108, 30, 1:3);
            for r = 1:numel(out)
                tc.verifyTrue(out(r).same_sign, out(r).label);
                tc.verifyLessThan(out(r).rel_err, 0.2, out(r).label);
            end
        end

        function optionValidation(tc)
            d = {{0}; {1}};
            e = @(ex) em_lsim_discriminative(d, 1, 1, 1, ex);
            % MMI is the only objective; any other objective raises a clear error
            for bad = {'ml', 'other_objective', 'MMI'}
                tc.verifyError(@() e(struct('objective', bad{1})), 'em_lsim_discriminative:badObjective', bad{1});
            end
            % options that are not MMI options are rejected, never ignored
            for unknown = {'not_an_option', 'learning_rate', 'eta', 'score_normalization', 'h'}
                tc.verifyError(@() e(struct('objective', 'mmi', unknown{1}, 1)), 'em_lsim_discriminative:unknownOption', unknown{1});
                tc.verifyError(@() e(struct(unknown{1}, 1)), 'em_lsim_discriminative:unknownOption', unknown{1});
            end
            tc.verifyError(@() e(struct('objective', 'mmi', 'mmi_recipe', 'legacy', 'ismooth_tau', 10)), ...
                'em_lsim_discriminative:publishedOptionOnly');
            tc.verifyError(@() e(struct('objective', 'mmi', 'damping_E', 3)), 'em_lsim_discriminative:legacyOptionOnly');
            tc.verifyError(@() e(struct('objective', 'mmi', 'mmi_recipe', 'current')), 'em_lsim_discriminative:badOptionValue');
            tc.verifyError(@() e(struct('objective', 'mmi', 'score_scale', -1)), 'em_lsim_discriminative:badOptionValue');
            o = lsim_disc_options(struct('objective', 'mmi'));
            tc.verifyEqual({o.mmi_recipe, o.score_scale, o.ebw_E, o.ismooth_tau, o.ismooth_backoff, o.ismooth_tau_discrete}, ...
                {'published', 1, 2, 100, 'ml', 0});
            tc.verifyEqual(lsim_disc_options(struct()), o, 'the default objective is mmi');
            tc.verifyEqual(lsim_disc_options(), o);
        end
    end
end

% =====================================================================
function Dmin = bisect_dmin(mu, v, n, s, r)
okf = @(D) all(n + D > 0) && all((r + D*(v + mu.^2))./(n + D) - ((s + D*mu)./(n + D)).^2 > 0);
if okf(0), Dmin = 0; return; end
lo = 0; hi = 1;
while ~okf(hi), hi = 2*hi; end
for it = 1:200
    mid = (lo + hi)/2;
    if okf(mid), hi = mid; else, lo = mid; end
end
Dmin = hi;
end
