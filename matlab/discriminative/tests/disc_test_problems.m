function varargout = disc_test_problems(kind, varargin)
%DISC_TEST_PROBLEMS Small fixed-seed problems shared by the unit tests.
%
%   [tr, init] = disc_test_problems('lsim', seed [, T])     C = 2 LSIM, 2 classes differing in dynamics
%   [tr, init] = disc_test_problems('hmm', seed [, M])      C = 1 HMM (exact score), M classes
%   [L, label, T] = disc_test_problems('scores', N, M)      random score matrix
%   [p, num, den, dom] = disc_test_problems('counts', seed [, den_scale])
%                       a C = 2 model (d = [1 2]) with real numerator / denominator counts under it
%                       and a loose domain (no projection binds)
%   q = disc_test_problems('em_from_counts', p, num)        the EM M-step of the counts
%   d = disc_test_problems('max_block_diff', a, b)          largest parameter difference
switch kind
    case 'lsim'
        seed = varargin{1};
        T = 30;
        if numel(varargin) > 1, T = varargin{2}; end
        rng(seed, 'twister');
        S = [2 2]; d = [1 1]; K = [1 1];
        t1 = disc_random_model(S, d, K, struct('self_coupling', 0.3));
        t2 = disc_random_model(S, d, K, struct('self_coupling', 0.3));
        t2.gmm_para_lsim = t1.gmm_para_lsim;
        varargout = {{disc_sample(t1, 12, T); disc_sample(t2, 12, T)}, {t1; t2}};
    case 'hmm'
        seed = varargin{1};
        M = 2;
        if numel(varargin) > 1, M = varargin{2}; end
        rng(seed, 'twister');
        tr = cell(M, 1);
        init = cell(M, 1);
        for k = 1:M
            t = disc_random_model(3, 2, 2);
            tr{k} = disc_sample(t, 6, randi([10 30], 6, 1), 0.05);
            init{k} = disc_random_model(3, 2, 2);
        end
        varargout = {tr, init};
    case 'scores'
        [N, M] = varargin{:};
        label = [1:M, randi(M, 1, N-M)]';
        label = label(randperm(N));
        T = randi([5 40], N, 1);
        L = -10*rand(N, M).*T/10 - 5;
        varargout = {L, label, T};
    case 'counts'
        seed = varargin{1};
        den_scale = 0.3;
        if numel(varargin) > 1, den_scale = varargin{2}; end
        rng(seed, 'twister');
        m = disc_random_model([2 3], [1 2], [1 1], struct('self_coupling', 0.3));
        obs = disc_sample(m, 8, 25);
        data = lsim_disc_prepare_data({obs; obs});
        p = lsim_disc_pack(m);
        N = numel(data.Y);
        wn = (data.label == 1).*(1 + rand(N, 1));
        wd = den_scale*rand(N, 1);
        [num, den] = lsim_disc_weighted_counts(p, data, wn, wd, lsim_fb_constants());
        dom = lsim_disc_domain('build', data, p, struct('prob_floor', 1e-300, 'mean_margin', 1e6, ...
            'var_lo_frac', 0, 'var_hi_frac', 1e12));
        varargout = {p, num, den, dom};
    case 'em_from_counts'
        [p, num] = varargin{:};
        q = p;
        sni = p.state_numbers_index;
        nrm = @(x) x/sum(x);
        for z = 1:p.C
            zi = sni(z)+1:sni(z+1);
            q.pi_0(zi) = nrm(num.pi(zi));
            for i = 1:sni(end)
                q.transitions_matrices(i, zi) = nrm(num.A(i, zi));
            end
            q.coupling_tetha_IM(:, z) = nrm(num.theta(:, z));
            di = p.dimension_numbers_index(z)+1:p.dimension_numbers_index(z+1);
            for i = 1:p.channel_num_states(z)
                q.P_all(z, i, 1) = 1;
                mu = num.s(di, i, 1)./num.n(di, i, 1);
                q.mu_all(di, i, 1) = mu;
                q.sigma_all(di, i, 1) = num.r(di, i, 1)./num.n(di, i, 1) - mu.^2;
            end
        end
        varargout = {q};
    case 'max_block_diff'
        [a, b] = varargin{:};
        f = {'pi_0', 'transitions_matrices', 'coupling_tetha_IM', 'P_all', 'mu_all', 'sigma_all'};
        d = 0;
        for j = 1:numel(f)
            d = max(d, max(abs(a.(f{j})(:) - b.(f{j})(:))));
        end
        varargout = {d};
    otherwise
        error('disc_test_problems: unknown kind %s', kind);
end
end
