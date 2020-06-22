function [ H ] = ProbHermitePolynomial( n,x )
%returns the n first probabilist hermite polynomials evaluated at x.
H=ones(n,length(x));
%Hn+1=xHn(x)-nHn-1(x)
%H(n+1)=Hn
H(2,:)=x;%by definition
for i=3:n
    H(i,:)=x.*H(i-1,:)-(i-1)*H(i-2,:);
end

