function [ cost ] = CostFuncNDF( mu_sigma)
%Cost function for lsqnonlin in Necula, Drimus, and Farkas setting
global NDF_call_fit;  global S; global K;global t;global r;global a;


cost=NDF_call_fit-Call_NDF( S,K,mu_sigma(1),a,mu_sigma(2),0,r,t,length(a));

end

