function [pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, AIC, log_likelyhood, BIC, pi_steady] = em_chmm( channels_observations , channel_num_states , num_gmm_component , max_itration , extra, init_model)
%EM_CHMM  Maximum-likelihood EM for a full coupled HMM (CHMM) with GMM emissions.
%
%   [pi_0_chmm, transition_matrices_chmm, chmm_gmm_para, AIC, log_likelyhood, BIC, pi_steady] = ...
%       em_chmm(channels_observations, channel_num_states, num_gmm_component, max_itration, extra, init_model)
%
% A CHMM conditions every channel's next state on the JOINT previous state of all
% channels, so its transition model has prod(M) rows per channel and exact inference
% runs on the equivalent single-chain HMM over prod(M) joint states. Use it for small
% models; em_lsim fits the LSIM, whose size grows with sum(M)^2 instead.
%
% em_chmm is em_lsim with the transition model replaced: the options, initialization,
% emission update, convergence check and choice of the returned iterate are em_lsim's,
% and the outputs come in em_lsim's order without its coupling matrix:
%   em_lsim: [pi_0, coupling, transitions, gmm, AIC, log_likelihood, BIC, pi_steady]
%   em_chmm: [pi_0,           transitions, gmm, AIC, log_likelihood, BIC, pi_steady]
% The returned parameters can be passed back as init_model (see below).
%
% channels_observations : C x N cell, {c,n} is the d(c) x T_n observation of channel c in
%                         sequence n (NaN = missing)
% channel_num_states    : C x 1 hidden states per channel, M(c)
% num_gmm_component     : C x 1 Gaussian components per state
% max_itration          : maximum number of EM iterations
% extra (optional)      : .sigma_diag (1 diagonal [default], 0 full covariance)
%                         .check_convergence (0 [default]: run all iterations)
%                         .left2right (0 [default]), .plot (0 [default])
%                         .iter_hook (optional @(itr, params, loglik, AIC, BIC) callback)
% init_model (optional) : start from given parameters instead of the GMM initialization:
%                         .pi_0_chmm (C x 1 cell, M(c) x 1), .transition_matrices_chmm
%                         (C x 1 cell, prod(M) x M(c), rows sum to one) and .chmm_gmm_para
%                         (the emission format below)
%
% transition_matrices_chmm{zee} is prod(M) x M(zee): row j is P(x^zee_t | joint state j at
% t-1). Joint states are ordered as in chmm_cartesian_product (the last channel varies
% fastest). pi_0_chmm{zee} is channel zee's initial distribution; the joint initial
% distribution is their product.
% chmm_gmm_para{zee}.gmm_para(i) holds state i's mixture: .P (K x 1 weights),
% .mu(k).x (d x 1 mean) and .sigma(k).x (d x 1 diagonal or d x d full covariance).
%
% AIC, log_likelyhood, BIC : per-iteration traces (exact log-likelihood of the parameters
% before that iteration's update). The returned parameters are the iterate with the
% highest log-likelihood, chosen as in em_lsim. pi_steady{zee} is the mean posterior
% state occupancy of channel zee.

if nargin < 5 || isempty(extra)
    extra = struct();
end
if isfield(extra,'sigma_diag')
    sigma_diag = extra.sigma_diag;
else
    sigma_diag = 1;
end
if ~isfield(extra,'left2right')
    extra.left2right = 0;
end
if ~isfield(extra,'plot')
    extra.plot = 0;
end
if ~isfield(extra,'check_convergence')
    extra.check_convergence = 0;
end

if(extra.plot==1)
    figure
    drawnow
    pause(0.01)
end

channel_num_states = channel_num_states(:);
num_gmm_component = num_gmm_component(:);

%Initializing all parameters
C = size(channels_observations , 1);
dim_observation = zeros(C , 1);
for zee = 1:C
    dim_observation(zee,1)  = size( channels_observations{zee,1} ,1);
end

state_numbers_index =  ( [0;cumsum(channel_num_states(:))] );
num_joint_states = prod(channel_num_states);

transition_matrices_chmm_packed = ones( num_joint_states , sum(channel_num_states)  );
pi_0_packed = zeros( sum(channel_num_states) , 1  );
P_all = zeros( C , max(channel_num_states) , max(num_gmm_component)  );
mu_all = zeros(  sum(dim_observation)  , max(channel_num_states) , max(num_gmm_component)   );
if sigma_diag
    sigma_all =  ones(  sum(dim_observation)  , max(channel_num_states) , max(num_gmm_component) );
else
    sigma_all =  ones(  sum(dim_observation)  , max(dim_observation) , max(channel_num_states) , max(num_gmm_component) );
end

num_trails = size(channels_observations,2);
length_observation(1,1)= length(channels_observations{1,1}(1,:));
for i=2:num_trails
    length_observation(i,1)= length(channels_observations{1,i}(1,:));
end
T_all = sum(length_observation);
all_observation =  zeros( sum(dim_observation) , T_all  );
dimension_numbers_index =  ( [0;cumsum(dim_observation(:))] );
trail_time_index =  ( [0;cumsum(length_observation(:))] );

for zee = 1:C
    Y=channels_observations{zee,1};
    for i=2:num_trails
        Y = cat(2, Y, channels_observations{zee,i});
    end
    all_observation(dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1),:)=Y;
end

if nargin == 6
    for zee = 1:C
        gmm_para = init_model.chmm_gmm_para{zee,1}.gmm_para ;
        for i=1:channel_num_states(zee)
            for k=1:num_gmm_component(zee)
                P_all(zee,i,k) =  gmm_para(i).P(k,1);
                if sigma_diag
                    sigma_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i , k) = gmm_para(i).sigma(k).x ;
                else
                    sigma_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1), 1:dim_observation(zee), i , k) = gmm_para(i).sigma(k).x ;
                end
                mu_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i , k) =  gmm_para(i).mu(k).x;
            end
        end
        if extra.left2right ==0
            temp_PI_0 = 0.0001+init_model.pi_0_chmm{zee,1};
        else
            temp_PI_0 = init_model.pi_0_chmm{zee,1};
        end
        pi_0_packed(state_numbers_index(zee)+1:state_numbers_index(zee+1) , 1) = temp_PI_0/sum(temp_PI_0);
        transition_matrices_chmm_packed( : , state_numbers_index(zee)+1:state_numbers_index(zee+1) ) = init_model.transition_matrices_chmm{zee,1};
    end
else
    for zee = 1:C
        Y = all_observation(dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1),:);
        % Initialization of emission probabilities (as in em_lsim)
        disp('Initializing: ')
        max_itr = 5;
        replicates_number = 20;
        init_gmm_numbers = channel_num_states(zee)*num_gmm_component(zee);
        covariance_type = 'diagonal';
        warning('off')
        gmm_para  = fine_gmm_fitting( Y, init_gmm_numbers, max_itr, replicates_number, covariance_type);
        warning('on')
        temp_mu = gmm_para.x.mu;
        temp_sigma = gmm_para.x.Sigma;
        P = gmm_para.x.ComponentProportion;
        for i=1:channel_num_states(zee)
            mu_state = temp_mu(1 ,:)';
            mu_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i , 1 ) = temp_mu(1 ,:)';
            if sigma_diag
                sigma_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i , 1) = squeeze(temp_sigma(1,:,1));
            else
                sigma_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1), 1:dim_observation(zee), i , 1) = diag(squeeze(temp_sigma(1,:,1)));
            end
            P_all(zee , i , 1) = P(1);
            P(1)=[];
            temp_mu(1 ,:)=[];
            temp_sigma(:,:,1)=[];
            for k=2:num_gmm_component(zee)
                temp_mu_gmm = mu_state(:,1);
                temp_distance = sum( abs(repmat(temp_mu_gmm,1,size(temp_mu,1)) - temp_mu').^2 );
                [~,ind_min]=min(temp_distance);
                mu_all(  dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i ,  k) = temp_mu(ind_min ,:)';
                if sigma_diag
                    sigma_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1) ,  i , k) = squeeze(temp_sigma(1,:,ind_min));
                else
                    sigma_all( dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1), 1:dim_observation(zee), i , k) = diag(squeeze(temp_sigma(1,:,ind_min)));
                end
                P_all(zee , i , k)=P(ind_min);
                P(ind_min)=[];
                temp_mu(ind_min ,:)=[];
                temp_sigma(:,:,ind_min)=[];
            end
            P_all(zee , i , :)= P_all(zee , i , :)./( sum( P_all(zee , i , :) ,3)  );
        end
        % uniform initial and transition probabilities
        pi_0_packed(state_numbers_index(zee)+1:state_numbers_index(zee+1) , 1) = ones(channel_num_states(zee) ,1  )/channel_num_states(zee);
        transition_matrices_chmm_packed( : , state_numbers_index(zee)+1:state_numbers_index(zee+1) ) =  1/channel_num_states(zee);
    end
end

P_all_new = P_all;
mu_all_new = mu_all;
sigma_all_new =  sigma_all;
transition_matrices_new = transition_matrices_chmm_packed;

alpha_T_trails = zeros( sum(channel_num_states) , T_all  );
P_observ_cond_to_state_trails = zeros( sum(channel_num_states) , T_all  );
P_observ_cond_to_state_comp_trails = zeros( sum(channel_num_states) , max(num_gmm_component) , T_all  );
gamma_trail_time_index = 1+(trail_time_index(1:end-1));
P_O_model =  zeros( 1 , num_trails );

% number of free parameters (AIC/BIC): CHMM transitions, emissions, mixture weights
if sigma_diag
    num_emission_parameters = sum(channel_num_states.*num_gmm_component*2.*dim_observation);
else
    num_emission_parameters = sum(channel_num_states.*num_gmm_component.*(dim_observation+dim_observation.^2));
end
num_parameters = sum(channel_num_states-1)*num_joint_states + num_emission_parameters + ...
    sum(channel_num_states.*(num_gmm_component-1));

for itr = 1:max_itration

    % ---- E-step: exact inference on the joint-state chain
    [pi_0_cell, transition_cell, gmm_cell] = unpack_chmm(pi_0_packed, transition_matrices_chmm_packed, ...
        P_all, mu_all, sigma_all, channel_num_states, num_gmm_component, dim_observation, ...
        state_numbers_index, dimension_numbers_index, sigma_diag);
    sum3_zeta = 0;
    for tr = 1:num_trails
        tr_index = trail_time_index(tr)+1:trail_time_index(tr+1);
        y = cell(C, 1);
        for zee = 1:C
            y{zee} = all_observation(dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1), tr_index);
        end
        [pi_joint, transition_joint, log_B_joint, index_matrix, P_observ_cond_to_state_trails(:,tr_index), ...
            P_observ_cond_to_state_comp_trails(:,:,tr_index)] = chmm_joint_model(pi_0_cell, transition_cell, gmm_cell, y);
        [P_O_model(tr), gamma_joint, ~, xi_sum] = chmm_exact_forward_backward(pi_joint, transition_joint, log_B_joint);
        % per-channel posteriors: marginals of the joint posterior
        for c=1:C
            for s=1:channel_num_states(c)
                alpha_T_trails(state_numbers_index(c)+s, tr_index) = sum(gamma_joint(index_matrix(c,:)==s , :),1);
            end
        end
        sum3_zeta = sum3_zeta + xi_sum;   % sum_t P(x_{t-1}=j, x_t=j' | data)
    end

    % ---- M-step: initial and CHMM transition probabilities
    gamma_P0 = alpha_T_trails(: , gamma_trail_time_index);
    pi_0_new = mean(gamma_P0,2);

    joint_next = sum3_zeta./repmat(sum(sum3_zeta,2),1,num_joint_states);
    joint_next(isnan(joint_next)) = 1/num_joint_states;
    for zee = 1:C
        zee_index = state_numbers_index(zee)+1:state_numbers_index(zee+1);
        for s=1:channel_num_states(zee)
            transition_matrices_new(:, zee_index(s)) = sum(joint_next(:, index_matrix(zee,:)==s),2);
        end
    end

    % ---- M-step: emissions (the em_lsim update)
    for zee = 1:C
        zee_index = state_numbers_index(zee)+1:state_numbers_index(zee+1);
        dim_zee = dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1);
        for i = 1:channel_num_states(zee)
            temp_gamma_obs = zeros(num_gmm_component(zee) , T_all );
            for k = 1:num_gmm_component(zee)
                temp_gamma_obs(k,:) = alpha_T_trails( zee_index(i),: ) .* (squeeze( P_observ_cond_to_state_comp_trails(zee_index(i),k,:) )'./P_observ_cond_to_state_trails( zee_index(i),:) );
                temp_gamma_obs(k,isnan(temp_gamma_obs(k,:)))=0;
                Y_update = all_observation(dim_zee,:);
                weight_update = repmat(temp_gamma_obs(k,:),dim_observation(zee,1),1);
                weight_update(isnan(Y_update))=0;
                Y_update(isnan(Y_update))=0;
                if sum(temp_gamma_obs(k,:)>10^-8)<=3 || sum(sum(~isnan(weight_update.*Y_update)&weight_update>0))<2*dim_observation(zee,1)
                    weight_update = weight_update+1;
                end
                mu_all_new( dim_zee ,  i , k ) =  sum( weight_update.*Y_update , 2)./ sum( weight_update , 2);
                if sigma_diag
                    Recentering = Y_update - repmat(mu_all_new( dim_zee ,  i , k ) , 1 , size(Y_update,2));
                    sigma_all_new( dim_zee ,  i , k ) =  0.001*var(all_observation(dim_zee,:),[],2,'omitnan') + ...
                        diag( (weight_update.*Recentering)*Recentering')./sum( weight_update , 2) ;
                else
                    Recentering = all_observation(dim_zee,:) - repmat(mu_all_new( dim_zee ,  i , k ) , 1 , size(Y_update,2));
                    Recentering(isnan(Recentering)) = 0;
                    weight_update = repmat(temp_gamma_obs(k,:),dim_observation(zee,1),1);
                    if sum(temp_gamma_obs(k,:)>10^-8)<=3 || sum(sum(~isnan(weight_update.*Y_update)&weight_update>0))<2*dim_observation(zee,1)
                        weight_update = weight_update+1;
                    end
                    sig_temp = diag(0.001*var(all_observation(dim_zee,:),[],2,'omitnan')) + ((weight_update.*Recentering)*Recentering')/sum(weight_update(1,:));
                    sig_temp = (sig_temp+sig_temp')/2;
                    if sum(eig(sig_temp)<=0)>1
                        sig_temp = sig_temp + diag( -min(eig(sig_temp))+ 0.001*var(all_observation(dim_zee,:),[],2,'omitnan')) ;
                    end
                    sigma_all_new( dim_zee, 1:dim_observation(zee), i , k) = sig_temp;
                end
                if(sum(temp_gamma_obs(:)<0)>0||sum(isnan(temp_gamma_obs(:)))>0)||sum(isnan(mu_all_new(:)))>0
                    disp('Error: reducing the state or gmm numbers')
                end
            end
            P_all_new(zee , i , 1:num_gmm_component(zee) ) = sum(temp_gamma_obs+eps,2)/sum(temp_gamma_obs(:)+eps);
        end
    end

    pi_steady_packed = sum( alpha_T_trails ,2)/ T_all;
    pi_0_packed = pi_0_new;
    transition_matrices_chmm_packed = transition_matrices_new;
    P_all = P_all_new;
    mu_all = mu_all_new;
    sigma_all =  sigma_all_new;

    AIC(itr)= -2*sum(P_O_model) + 2*num_parameters;
    BIC(itr)= -2*sum(P_O_model) + log(T_all)*num_parameters;
    log_likelyhood(itr) = sum(P_O_model);

    if isfield(extra,'iter_hook') && ~isempty(extra.iter_hook)
        extra.iter_hook(itr, struct('pi_0', pi_0_packed, 'transition_matrices_chmm', transition_matrices_chmm_packed, ...
            'P_all', P_all, 'mu_all', mu_all, 'sigma_all', sigma_all), ...
            log_likelyhood(itr), AIC(itr), BIC(itr));
    end

    if(extra.check_convergence>0)
        if(itr>9)
            if(sum((diff(log_likelyhood(itr-5:itr)))<0.001)>=2)
                if(extra.plot==1)
                    disp('EM algorithm converged')
                end
                break;
            end
        end
    end

    % keep the iterate with the highest log-likelihood (as em_lsim)
    if(itr>2)
        if(max_loglik<log_likelyhood(end))
            max_loglik = log_likelyhood(end);
            P_all_max = P_all;
            mu_all_max = mu_all;
            sigma_all_max =  sigma_all;
            transitions_matrices_max = transition_matrices_chmm_packed;
            pi_0_max = pi_0_packed;
            pi_steady_max = pi_steady_packed;
        end
    else
        max_loglik = log_likelyhood(end);
        P_all_max = P_all;
        mu_all_max = mu_all;
        sigma_all_max =  sigma_all;
        transitions_matrices_max = transition_matrices_chmm_packed;
        pi_0_max = pi_0_packed;
        pi_steady_max = pi_steady_packed;
    end

    if(rem(itr,10)==0&&extra.plot==1)
        hold on
        plot(log_likelyhood,'m')
        drawnow
        pause(0.01);
    end
end

if(extra.plot==1)
    hold on
    plot(log_likelyhood,'m')
    pause(0.01);
    disp(['Itr: ',num2str(itr),', Log likelyhood: ',num2str(sum(P_O_model)),', AIC: ',num2str(AIC(end)),', BIC: ',num2str(BIC(end))])
end

% output parameters
[pi_0_chmm, transition_matrices_chmm, chmm_gmm_para] = unpack_chmm(pi_0_max, transitions_matrices_max, ...
    P_all_max, mu_all_max, sigma_all_max, channel_num_states, num_gmm_component, dim_observation, ...
    state_numbers_index, dimension_numbers_index, sigma_diag);
pi_steady = cell(C,1);
for zee = 1:C
    pi_steady{zee,1} = pi_steady_max(state_numbers_index(zee)+1:state_numbers_index(zee+1) , 1);
end

end

function [pi_0_cell, transition_cell, gmm_cell] = unpack_chmm(pi_0_packed, transition_packed, P_all, mu_all, sigma_all, ...
    channel_num_states, num_gmm_component, dim_observation, state_numbers_index, dimension_numbers_index, sigma_diag)
% packed arrays -> the cell format of chmm_cartesian_product / forward_backward_chmm
C = numel(channel_num_states);
pi_0_cell = cell(C,1);
transition_cell = cell(C,1);
gmm_cell = cell(C,1);
for zee = 1:C
    zee_index = state_numbers_index(zee)+1:state_numbers_index(zee+1);
    dim_zee = dimension_numbers_index(zee)+1:dimension_numbers_index(zee+1);
    gmm_para = [];
    for i=1:channel_num_states(zee)
        for k=1:num_gmm_component(zee)
            gmm_para(i).P(k,1) = P_all(zee,i,k); %#ok<AGROW>
            if sigma_diag
                gmm_para(i).sigma(k).x = sigma_all(dim_zee, i, k);
            else
                gmm_para(i).sigma(k).x = sigma_all(dim_zee, 1:dim_observation(zee), i, k);
            end
            gmm_para(i).mu(k).x = mu_all(dim_zee, i, k);
        end
    end
    gmm_cell{zee,1}.gmm_para = gmm_para;
    pi_0_cell{zee,1} = pi_0_packed(zee_index, 1);
    transition_cell{zee,1} = transition_packed(:, zee_index);
end
end

function gmm_para  = fine_gmm_fitting( inpute_features, gmm_numbers, max_itr, replicates_number, covariance_type)
% covariance_type is 'full' | 'diagonal' (the em_lsim initializer)
if nargin <5
    covariance_type = 'diagonal';
end
if nargin <4
    replicates_number = 1;
end
options = statset('Display','off','MaxIter',max_itr);
flag_check = 0;
counter = 0;
while(flag_check==0&&counter<10)
    counter = counter+1;
    try
        this_gmm_model = fitgmdist( inpute_features' , gmm_numbers ,'Start','randSample','Options', options,'Regularize',0.01*min(var(inpute_features,[],2,'omitnan')) , 'CovarianceType',covariance_type,'Replicates',replicates_number);
        BIC_P = this_gmm_model.BIC;
        flag_check=1;
    catch exception
        try
            if(strfind(exception.message,'Ill-conditioned'))
                disp('Requlized +1')
                this_gmm_model = fitgmdist( inpute_features' , gmm_numbers ,'Start','randSample','Options', options,'Regularize',0.01*min(var(inpute_features,[],2,'omitnan')) , 'CovarianceType',covariance_type,'Replicates',replicates_number);
                BIC_P = this_gmm_model.BIC;
                flag_check=1;
            end
        catch
            for i = 1:size(inpute_features,1)
                inpute_features(i,:) = inpute_features(i,:)  + counter*0.05*mean(std(inpute_features(i,:),[],2))*randn(size(inpute_features(i,:) ,1) , size(inpute_features(i,:) ,2));
            end
            disp('Not converge +1')
        end
    end
end
gmm_para.x =  this_gmm_model;
gmm_para.BIC =  BIC_P;
gmm_para.Num_Com = gmm_numbers;
end
