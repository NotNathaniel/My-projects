function [ q ] = qGenerator( N, c )
%generates the q-values according to p.57
%c is a vector of cumulants
q=zeros(N+1,1);%the 0th term is included
q(1)=1;
%we want q to be a column vector
if N<=5
    q(4:N+1)=c(3:N)./(factorial(1).*factorial(3:N)'.*sqrt(c(2)).^(3:N)');%n=3,4,5, m=1
else
    q(4:6)=c(3:5)./(factorial(1).*factorial(3:5)'.*sqrt(c(2)).^(3:5)');
end

if (N==6 || N==7)
    for i=7:N+1
    q(i)=c(i-1)./(factorial(1)*factorial(i-1)*sqrt(c(2))^(i-1));
    fn2=floor((i-1)/2);
    cn2=ceil((i-1)/2);

    q(i)=q(i)+2*prod(c([fn2 cn2]))/(factorial(2)*factorial(fn2)*factorial(cn2)*...
        sqrt(c(2))^(i-1));
    end
elseif(N==8)
    for i=7:8%n=6,7
        q(i)=c(i-1)./(factorial(1)*factorial(i-1)*sqrt(c(2))^(i-1));
        fn2=floor((i-1)/2);
        cn2=ceil((i-1)/2);
        q(i)=q(i)+prod(c([fn2 cn2]))/(factorial(2)*factorial(fn2)*factorial(cn2)*...
        sqrt(c(2))^(i-1));
    end
end
if N==8
    q(8)=c(8)/(factorial(8)*sqrt(c(2))^8)+c(4)^2/(factorial(2)*2*factorial(4)*sqrt(c(2))^8)+...
        c(5)*c(3)/(factorial(2)*factorial(5)*factorial(3)*sqrt(c(2))^8);
    
elseif N>8
    error('We will not go further than 8');
end


end