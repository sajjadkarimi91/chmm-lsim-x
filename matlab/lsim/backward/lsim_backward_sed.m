function [beta , alpha_T] = lsim_backward_sed( alpha , b_c_ot_nc , transitions_matrices , coupling_tetha_IM , layout , fb_constants )
%LSIM_BACKWARD_SED  SED backward recursion of the LSIM forward-backward ('sed').
%
%   [beta, alpha_T] = lsim_backward_sed(alpha, b_c_ot_nc, A, theta, layout, fb_constants)
%
% Backward message built from the pairwise conditionals P(v^zee_t-1, v^w_t | o) of the
% forward pass. For each channel the incoming backward messages of all channels are
% mixed with delta weights solved from
%   x = ( diag(f-h) + h*ones(C,C) ) \ f
% (closed form below), floored at fb_constants.beta_floor, with fb_constants.f_reg
% added to f. Stores sum(M) x sum(M) x T pairwise arrays, so it is the most
% memory-hungry built-in variant.
%
% Arguments and outputs follow the backward plug-in interface, see
% lsim/backward/Contents.m.

C = layout.C;
T = layout.T;
state_numbers_index = layout.state_numbers_index;
channel_num_states = layout.channel_num_states;
vec_mat_mult = layout.vec_mat_mult;
mat_mult_one = layout.mat_mult_one;

alpha_T  = zeros( state_numbers_index(end) , T  );
beta  = zeros( state_numbers_index(end) , T );
mat_mult = zeros( state_numbers_index(end) , C  );

cuopling_repmat_beta = zeros( state_numbers_index(end) , C  );
coupling_transmat = zeros( state_numbers_index(end) , state_numbers_index(end)  );
diag_zero = ones( state_numbers_index(end) , state_numbers_index(end)  );

P_vz_tm1_vw_t_O = zeros(state_numbers_index(end) , state_numbers_index(end)  , T );
P_vz_tm1_vw_t_O_cond = zeros(state_numbers_index(end) , state_numbers_index(end)  , T );

for zee =  1:C

    zee_index = ( state_numbers_index(zee)+1:state_numbers_index(zee+1) );

    diag_zero( zee_index ,  zee_index )= 0 ;

    zee_coupling = coupling_tetha_IM(zee,:);
    cuopling_repmat_beta( zee_index ,: ) = repmat( zee_coupling ,channel_num_states(zee),1);

end

for zee = 1:C
    zee_index = ( state_numbers_index(zee)+1:state_numbers_index(zee+1) );
    coupling_transmat( : ,  zee_index ) = repmat(cuopling_repmat_beta(:,zee) , 1 , length(zee_index) ).* transitions_matrices(: , zee_index);
end

% pairwise conditionals from the forward pass
for t = (2:T)

    alpha_tm1_tm1 = alpha(: , t-1).*b_c_ot_nc(:,t-1);

    P_vz_tm1_vw_t_O_cond(:,:,t) =  ( coupling_transmat + (diag_zero.*repmat( alpha_tm1_tm1' , state_numbers_index(end)  , 1 )) * coupling_transmat );
    P_vz_tm1_vw_t_O(:,:,t) = P_vz_tm1_vw_t_O_cond(:,:,t).*(repmat( alpha_tm1_tm1 , 1 , state_numbers_index(end) )) ;

end


beta(: , T) = b_c_ot_nc(: , T);
alpha_T(: , T) =  alpha(: , T).*beta(: , T);

flag_nan = sum(sum(coupling_tetha_IM,2)<=10^(-10))>0;

channel_numbers_index = [0,C:C:C^2];

%Compute Backward variables
for t = (T-1:-1:1)

    mat_mult(vec_mat_mult) = beta(: , t+1);
    beta_C = P_vz_tm1_vw_t_O_cond(:,:,t+1)*mat_mult ;%compute from t=1 to t=T-1
    %weighted average beta

    if(C>1)

        temp = P_vz_tm1_vw_t_O(:,:,t+1);
        % alpha(:,t+1) can be exactly 0 for a state with no incoming mass
        % (sparse/left-to-right transitions). temp(:,j) is then exactly 0
        % too for that j (both are sums of the same nonnegative products),
        % so the true contribution to f is 0, not 0/0=NaN. Divide only
        % where the denominator is nonzero and leave those columns at 0.
        alpha_next_row = alpha(: , t+1)';
        denom_2d_f = repmat(alpha_next_row, state_numbers_index(end),1 );
        temp_2d_f = zeros(size(temp));
        nz_cols = denom_2d_f(1,:) ~= 0;
        temp_2d_f(:,nz_cols) = (temp(:,nz_cols).^2)./denom_2d_f(:,nz_cols);
        temp_aplha =  alpha(: , t).* b_c_ot_nc( : , t);

        h_zee_temp =  cumsum(temp_aplha.^2);
        h_zee_all = h_zee_temp(state_numbers_index(2:end)) - [0;h_zee_temp(state_numbers_index(2:end-1))] ;
        h_zee_all = repelem(h_zee_all,C,1);

        f_zee_temp = cumsum(temp_2d_f,1);
        f_zee_temp = f_zee_temp(state_numbers_index(2:end),:) - [zeros(1,state_numbers_index(end));f_zee_temp(state_numbers_index(2:end-1),:)] ;
        f_zee_temp = cumsum(f_zee_temp,2);

        f_zee_all = ( f_zee_temp(:,state_numbers_index(2:end)) - [zeros(C,1),f_zee_temp(:,state_numbers_index(2:end-1))] )';
        f_zee_all = f_zee_all(:)+fb_constants.f_reg;

        % x = ( diag(f-h) + h*ones(C,C) ) \ f in closed form (Sherman-Morrison):
        % delta(w) = ( f(w) - g )/( f(w) - h ),
        % g = h*sum_v f(v)/(f(v)-h) / (1 + h*sum_v 1/(f(v)-h))
        nume_temp = cumsum(f_zee_all./(f_zee_all-h_zee_all));
        nume_temp = nume_temp(channel_numbers_index(2:end)) - [0;nume_temp(channel_numbers_index(2:end-1))] ;
        nume_temp = h_zee_all.*repelem(nume_temp,C,1);

        denume_temp = cumsum(1./(f_zee_all-h_zee_all));
        denume_temp = denume_temp(channel_numbers_index(2:end)) - [0;denume_temp(channel_numbers_index(2:end-1))] ;
        denume_temp =1+ h_zee_all.*repelem(denume_temp,C,1);

        g_zee_all = nume_temp./denume_temp;

        d_hat = (f_zee_all-g_zee_all)./(f_zee_all-h_zee_all);

        temp_delta = reshape(d_hat ,C,C)';

        % delta rows are not normalized: a constant row scale cancels in the
        % per-channel normalization of beta and alpha_T below

        mat_mult(vec_mat_mult) = 1;
        weights_subsystems = mat_mult * temp_delta;

    else
        mat_mult(vec_mat_mult) = 1;
        weights_subsystems = mat_mult;
    end

    beta(: , t) = b_c_ot_nc( : , t).*( max(fb_constants.beta_floor, sum( beta_C.*weights_subsystems  , 2) ) ) ;

    if(flag_nan)
        beta(isnan(beta(: , t)) , t)=  b_c_ot_nc( isnan(beta(: , t)) , t);
    end

    alpha_T( : , t) = beta(: , t).*alpha(: , t);

    temp_sum = cumsum(alpha_T( : , t));
    normalizer = temp_sum(state_numbers_index(2:end))-[0;temp_sum(state_numbers_index(2:end-1))];
    normalizer_vec = mat_mult_one*normalizer(:);

    beta( : , t) = beta( : , t)./normalizer_vec;
    alpha_T( : , t) = alpha_T( : , t)./normalizer_vec;

end

end
