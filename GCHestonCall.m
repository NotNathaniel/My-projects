function [ call, put] = GCHestonCall( c,k,N,J, I, H,q )
%c - vector of cumulants
%k - strike price
%N - length of c
%J - vector of J-values as described in Yin's paper
%H - vector of probabilist Hermite polynomials
%q - vector of q's
%q,H,J, are all N+1

%n x length(k) .* n x 1 matrices

x=(k-c(1))/sqrt(c(2));

vec=[2:N-1]';

if(N~=3)
    first=exp(c(1))*sum(J.*q);%sums over columns

    second=exp(k).*[normcdf((c(1)-k)/sqrt(c(2)))+sum(...
        q(4:end).*H(3:end-1,:)).*normpdf((k-c(1))/sqrt(c(2)))];
    
    firstp=exp(k).*[normcdf(x)-sum(q(4:end).*H(3:end-1,:).*normpdf(x))];
    
    secondp=exp(c(1))*sum(I.*q);

else
    first=exp(c(1))*sum(J.*q);
    second=exp(k).*[normcdf((c(1)-k)/sqrt(c(2)))+...
        q(end).*H(end-1,:).*normpdf((k-c(1))/sqrt(c(2)))];
    
    firstp=exp(k).*[normcdf(x)-q(4:end).*H(end-1,:).*normpdf(x)];
    secondp=exp(c(1))*sum(I.*q);
end
call=first-second;%should be a 1x length(k) vector
put=firstp-secondp;



%c+Kexp(-r*t)-p-S=-0.284*1e-04 vector of all these.... put call parity
%is satisfied

end

