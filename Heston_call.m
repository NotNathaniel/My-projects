function [ price ] = Heston_call( params, S, K, t, lambda, r )
%Computes call option price using same parameters as Heston, 1994
kappa=params(1);
theta=params(2);
sigma=params(3);
v0=params(4);
rho=params(5);
x=log(S);
a=kappa*theta;
b=[kappa+lambda-rho.*sigma; kappa+lambda];
u=[1/2 ;-1/2];
d1=@(phi) sqrt((rho*sigma.*phi.*1i-b(1)).^2-sigma.^2.*(2.*u(1).*phi.*1i-phi.^2));
g1=@(phi) (b(1)-rho.*sigma.*phi.*1i+d1(phi))/(b(1)-rho.*sigma.*phi.*1i-d1(phi));
%what is d(phi)?

C1=@(phi) r.*phi.*1i.*t+a./sigma.^2.*[(b(1)-rho.*sigma.*phi.*1i+d1(phi)).*t-2.*log((1-g1(phi).*exp(d1(phi).*t))./(1-g1(phi)))];
D1=@(phi) (b(1)-rho.*sigma.*phi.*1i+d1(phi))./sigma.^2.*(1-exp(d1(phi).*t))/(1-g1(phi).*exp(d1(phi).*t));


d2=@(phi) sqrt((rho*sigma.*phi*1i-b(2)).^2-sigma^2.*(2.*u(2).*phi.*1i-phi.^2));
g2=@(phi) (b(2)-rho.*sigma.*phi.*1i+d2(phi))./(b(2)-rho.*sigma.*phi.*1i-d2(phi));
%what is d(phi)?
C2=@(phi) r.*phi.*1i.*t+a/sigma.^2.*[(b(2)-rho.*sigma.*phi.*1i+d2(phi)).*t-2.*log((1-g2(phi).*exp(d2(phi).*t))./(1-g2(phi)))];
D2=@(phi) (b(2)-rho.*sigma.*phi.*1i+d2(phi))/sigma.^2.*(1-exp(d2(phi).*t))/(1-g2(phi).*exp(d2(phi).*t));

%integral doesn't take in cell-arrays...
f1=@(phi) exp(C1(phi)+D1(phi).*v0+1i.*phi.*x);

f2=@(phi) exp(C2(phi)+D2(phi).*v0+1i.*phi.*x);
l=@(phi) real((exp(-1i*phi)*f1(phi))/(1i*phi));
P1=1/2+1/pi*integral(@(phi) real((exp(-1i.*phi).*f1(phi))./(1i.*phi)),0,50);

P1=1/2+1/pi*integral(@(phi) real((exp(-1i.*phi.*log(K)).*f1(phi))./(1i.*phi)),0,100,'ArrayValued',true);

P2=1/2+1/pi*integral(@(phi) real((exp(-1i.*phi.*log(K)).*f2(phi))./(1i.*phi)),0,100,'ArrayValued',true);

price=S.*P1-K*exp(-r*t).*P2;
