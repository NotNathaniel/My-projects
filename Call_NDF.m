function [ call ] = Call_NDF( S,K,mu,a,sigma,qdiv,r,t,n)
%computes call options in the NDF setting with the same parameter set
%as described in Necula, Drimus, and Farkas.

d1J=[log(S./K)+(mu+sigma^2)*t]/[sigma*sqrt(t)];
d2J=d1J-sigma*sqrt(t);%this d2 is for f

H=PhysHermitePolynomial(n,-d2J);
[J,I]=JI_GeneratorNDF(d1J,d2J,H,n,sigma,t );

SP=S*exp((mu-r+sigma^2/2)*t);
Kt=K*exp(-r*t);

Pi1=exp((mu-(r-qdiv)+sigma^2/2)*t)*sum(a.*I);%multiplied by right elements
Pi2=sum(a.*J);

call=S*exp(-qdiv*t)*Pi1-K.*exp(-r*t).*Pi2;

end

