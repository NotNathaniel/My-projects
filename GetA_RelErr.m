function [ a,mu] = GetA_RelErr( S,K,t, r,qdiv,sigma,data_call,n,a ) 

mu=r+0.01;
d1J=[log(S./K)+(mu+sigma^2)*t]/[sigma*sqrt(t)];
d2J=d1J-sigma*sqrt(t);%this d2 is for f

H=PhysHermitePolynomial(n,-d2J);
[J,I]=JI_GeneratorNDF(d1J,d2J,H,n,sigma,t );

SP=S*exp((mu-r+sigma^2/2)*t);
Kt=K*exp(-r*t);
H=zeros(n+1);%no longer hermite

Ssum=sum(I./data_call,2);
KJsum=sum(Kt.*J./data_call,2);


f=-2*[SP*Ssum-KJsum; -1/2*length(data_call)];%n+1x1 vector

for i=1:length(I(:,1))
    for j=1:length(I(:,1))
        H(i,j)=SP^2*sum(I(i,:).*I(j,:)./data_call.^2)-2*SP*sum(Kt.*I(i,:).*J(j,:)./data_call.^2)+...
            sum(Kt.^2.*J(i,:).*J(j,:)./data_call.^2);
    end
end

H=2*H;%because we want all of H involved
H=(H+H')/2;%Make H symmetric


N=5000;


lb=-4*ones(length(f)-1,1)./(15.^[0:1/2:(length(f)-2)/2]');%0.0084 data


ub=[-lb];
lb=[lb;1];
ub=[ub;1];



options=optimoptions('fmincon','Algorithm','active-set','MaxIterations',1300,'MaxFunctionEvaluations',500, 'ScaleProblem','obj-and-constr',...
    'FiniteDifferenceType', 'central', 'StepTolerance',1e-14,'OptimalityTolerance',1e-16 );%fmincon or fminunc
%options=optimoptions('fmincon','Display','iter');

[A,Aeq,b,beq]=Linprog_matrices( n,mu,sigma,N);
Aeq=[zeros(1,n),1];
beq=1;

a=fmincon(@(x) x'*H*x/2+f'*x,[a(1:length(f)-1);1],A,b,Aeq,beq,lb,ub,[]);
%no non-linear constraints
a=a;
a(end)=[];%removing element for call option prices




