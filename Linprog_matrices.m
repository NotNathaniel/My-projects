function [ A,Aeq,b,beq ] = Linprog_matrices( m,mu,sigma,N)
%m-number of coefficients
%N - length of grid
%H - mx1 vector of hermite polynomials, recall that these hermite
% n- number of strike prices
%yin_call - 1xn vector of call prices
%polynomials don't care about K
N=2*round(sqrt(N));
x=sin(linspace(-pi/2,pi/2,N+1));
x=6*m*x-3*m;
x=x';
%x=[-3*m:6*m/N:3*m]';%N+1x1
%regenerate H
H=zeros(N+1,m);
H(:,1)=ones(N+1,1);
H(:,2)=2*(x-mu)/sigma;
for i=3:m
    H(:,i)=2.*(x-mu)/sigma.*H(:,i-1)-2*(i-1)*H(:,i-2);
end
A=[-1/sigma*normpdf((x-mu)/sigma).*H,zeros(N+1,1)];%column vector times row vector
%^probability density

evenvec=[0:2:m-1];%we go to m-1


evenvec=factorial(2*evenvec)./factorial(evenvec);%astronomical values

if (mod(m-1,2)==1)%last coefficient has an odd subscript then we have
    %as many 0s as elements
    unit_mass=reshape([evenvec;zeros(size(evenvec))],1,[]);
else
    unit_mass=reshape([evenvec;zeros(1,length(evenvec))],1,[]);
    unit_mass(end)=[];%removing last 0
end

Aeq=[zeros(1,m),1];

Aeq=[Aeq;[unit_mass,0]];

beq=[1;1];

b=[zeros(N+1,1)];


end

