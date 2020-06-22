function [ cost ] = CostFuncBS( sigmaBS)
%Cost function for lsqnonlin in Black Scholes setting
global call_price;  global S; global K;global t;global r;

cost=call_price-blsprice(S,K,r,t,sigmaBS);

end

