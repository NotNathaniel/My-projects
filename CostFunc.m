function [ cost ] = CostFunc( params)
%Cost function for lsqnonlin in Heston setting (also used for Gram-Charlier
%setting
global call_price;  global S; global K;global t;global r;

cost=call_price-Heston_call(params,S,K,t,0,r);%lambda=0

end

