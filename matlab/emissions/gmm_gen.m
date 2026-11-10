function x = gmm_gen( gmm_param , state_number )
%GMM_GEN  Draw one observation from the Gaussian mixture of a hidden state.
%
%   x = gmm_gen(gmm_param, state_number)
%
% gmm_param is a channel's mixture array (gmm_para_lsim{c}.gmm_para): .P (K x 1
% weights), .mu(k).x (d x 1 mean), .sigma(k).x (d x 1 diagonal or d x d full
% covariance). Returns a d x 1 sample of state state_number, drawn the same way
% as in generate_chmm_time_series.

P = gmm_param(state_number).P;
n = select_p( P );

if size(gmm_param(1).sigma(1).x,2)==1
    x = diag(gmm_param(state_number).sigma(n).x)^0.5*randn(length(gmm_param(state_number).mu(n).x) , 1) + gmm_param(state_number).mu(n).x ;
else
    x = (gmm_param(state_number).sigma(n).x)^0.5*randn(length(gmm_param(state_number).mu(n).x) , 1) + gmm_param(state_number).mu(n).x ;
end

end


function x = select_p( p )

% index drawn from the discrete distribution p (sum(p) = 1)

temp = find(cumsum(p) > rand(1,1));
x = temp(1);

end
