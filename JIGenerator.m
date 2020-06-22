function [ J, I] = JIGenerator( k,c,n,H,option )
%generates J values from Yin's paper p.60
%k-log strike
%c- vec of cumulants
%n=length(c)
J=ones(n,length(k));
I=ones(n,length(k));
x=(k-c(1))/sqrt(c(2));%there is no c0
a=sqrt(c(2));
J(1,:)=exp(a^2/2)*normcdf(a-x);%1st row
I(1,:)=exp(a^2/2)*normcdf(x-a);

phi=normpdf(x);%1 x nk
eax=exp(a*x);%1 x nk
for i=2:n
    J(i,:)=a.*J(i-1,:)+H(i-1,:).*phi.*eax;%for all the different K
    I(i,:)=a.*I(i-1,:)-H(i-1,:).*phi.*eax;
    %rows 2 to n, each iteration handles a full row.
end


end
%