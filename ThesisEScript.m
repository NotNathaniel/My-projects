%Script for comparing an option under the heston model priced by
%gram-charlier expansion vs one priced with a Gauss-Hermite related
%risk-neutral density.
hold off
clear
%Model fitting section

global S;
global r;
global t;
v0=0.03;kappa=0.15; sigma=0.05; rho=-0.55;t=1;
S=100;qdiv=0;

theta=0.05;r=0.04;x0=log(S)-r*t;%these calls vary extreme amounts for different n
%precomputed cumulants:
%c=Cumulant_Sym(...)
c=zeros(8,1);
c =[
    
4.589456320893087
0.031838851938017
-0.001261815533270
0.000115992103852
-0.000011185767429
0.000001414417922
-0.000000200913900
0.000000027458226];
%almost exact same when n=3
global K;
K=40:0.001:170;%can't have 0 since it causes NaN values
%you can't change this: its the save
K3=K;
k=log(K)-r*t;%1x101 vector

%here we will compare which method fits to the heston model best.
exact_calls=load('HestonCalls.txt');%computed by Heston_call
exact_calls=exact_calls';
exact_puts=exact_calls+K*exp(-r*t)-S;
x=(k-c(1))/sqrt(c(2));
qdiv=0;

rng(1);%seeding
c2=cvpartition(length(K),'KFold',3);%only need to fit data

%how well the models fit for different K
test_errorNDF=zeros(6,1);
train_errorNDF=zeros(6,1);
test_errorGC=zeros(6,1);
train_errorGC=zeros(6,1);

time_GC=zeros(6,1);
time_NDF=zeros(6,1);
an=zeros(14,6);
nancount=0;
ub=[1,2];
lb=[r,0];
S=1665.53;
qdiv=0;
t=231/365;
r=0.04;%https://ycharts.com/indicators/1_year_treasury_rate Nov, 2018

global NDF_call_fit;
global a;

%timing
tic

for i=1:c2.NumTestSets%get test and train error
    disp(['Fold ',num2str(i)])
    m=4;
    
    sigma=0.1759;
    mu=0.04;
    %initializing values from previous optimization setting
    init=[
        0.1173
        -0.0601
        0.0084
        0.0017
        0.0025
        0.0000
        0.0199
        0.0000
        0
        0
        0
        0
        0
        0]*1e-6;
    for n=3:8%qn
        tic
        q=qGenerator( n, c );
        H=ProbHermitePolynomial(n+1,x);
        [J,I]=JIGenerator( k,c,n+1,H );
        [yin_call, yin_put]=GCHestonCall( c,k,n,J,I,H,q );
        train_errorGC(n-2)=train_errorGC(n-2)+mean(abs(exact_calls(~c2.test(i))-yin_call(~c2.test(i)))./max(exact_calls(~c2.test(i)),0.01));
        test_errorGC(n-2)=test_errorGC(n-2)+mean(abs(exact_calls(c2.test(i))-yin_call(c2.test(i)))./max(exact_calls(c2.test(i)),0.01));
        time_GC(n-2)=toc;
        
        if (m<20)
            %fitting a-parameters
            if n>3
                [a,mu]= GetA_RelErr( S,K3(~c2.test(i)),t, r,qdiv,sigma,...
                    exact_calls(~c2.test(i)),m,an(1:m,n-3));
            else
                [a,mu]= GetA_RelErr( S,K3(~c2.test(i)),t, r,qdiv,sigma,...
                    exact_calls(~c2.test(i)),m,init);%;2./(10.^[0:1/4:(m-1)/4]')
                %GetA( S,K2,t, r,qdiv, sigma,fit_call,n,a_mat(1:n,index-1));
            end
            an(1:length(a),n-2)=a;%let a solution be your starting point!
            NDF_call_fit=exact_calls(~c2.test(i));%for the cost function
            %in the line where we find 'params'
            K=K3(~c2.test(i));
            params=lsqnonlin(@CostFuncNDF,[r+0.01,sigma],lb,ub);
            mu=params(1);
            sigma=params(2);
            call_NDF=Call_NDF( S,K3,mu,a,sigma,qdiv,r,t,m);
            
            train_errorNDF(n-2)=train_errorNDF(n-2)+mean(abs(exact_calls(~c2.test(i))-call_NDF(~c2.test(i)))./max(exact_calls(~c2.test(i)),0.01))
            test_errorNDF(n-2)=test_errorNDF(n-2)+mean(abs(exact_calls(c2.test(i))-call_NDF(c2.test(i)))./max(exact_calls(c2.test(i)),0.01));
            time_NDF(n-2)=toc;
            
        end
        m=m+2;
        
    end
end
total_time=toc;
test_errorNDF=test_errorNDF/c2.NumTestSets
test_errorGC=test_errorGC/c2.NumTestSets;

train_errorNDF=train_errorNDF/c2.NumTestSets;
train_errorGC=train_errorGC/c2.NumTestSets;

%% Data Section
%parameters
v0=0.03;kappa=0.15; sigma=0.05; rho=-0.55;
global S;
S=1665.53;
qdiv=0;
global t;
t=231/365;
global r;
%risk-free rate
r=0.027;%https://ycharts.com/indicators/1_year_treasury_rate Nov, 2018

theta=0.05;x0=log(S);
optprice=load('AmznDATA.txt');
%first is a call, last is a put
index=mod([1:length(optprice)],2)==0;
global K;
K=optprice(index,1)';
K3=K;%assigning all the K data points to this

put_price=optprice(index,2)';
global call_price;

call_price_full=optprice(~index,2)';%assigning all call price data points
call_price=optprice(~index,2)';
put_price_full=put_price;%this will be relevant later.

%from a previous fitting session
%these are parameters from the Heston model
params =[
    
-1.0784
0.2198
0.2507
0.1507
-0.3201];
kappa=params(1);theta=params(2);sigma=params(3);v0=params(4);rho=params(5);
n=3;

%c=Cumulant_Sym( v0,x0,kappa,sigma,rho,t,theta,n )
%these are the first n cumulants
c =[
    
7.3797
0.0794
-0.0103
];
%cubic splines
%GC-expansion fitting
K2=min(K):0.05:max(K);

k=log(K2)-r*t;
length(K)
length(call_price)
length(K2)
fit_call=spline(K,call_price,K2);
fit_put=spline(K,put_price,K2);
% q=qGenerator( n, c );%most q values are 0
% x=(k-c(1))/sqrt(c(2));%there is no c0, this is a 101x1 vector
% H=ProbHermitePolynomial(n+1,x);%n+1 since it starts at
% [J,I]=JIGenerator( k,c,n+1,H );
% [yin_call, yin_put]=GCHestonCall( c,k,n,J,I,H,q );


%GH risk-neutral density fitting
n=12;

%this bottom code yielded horrendous results
%call_NDF_clos=Call_NDF_closed( S,K,t, r,qdiv, mu,sigma,yin_call,yin_put,n );


%initial guess for sigma in NDF model
sigma=0.3;


global NDF_call_fit;
global call_price;
NDF_call_fit=call_price;
global K;
global a;

ub=[1,2];
lb=[r,0];
n_start=2;
n_incr=2;
n_max=16;
n_times=n_max/n_incr-(n_start-n_incr)/n_incr;
a_mat=zeros(n_max,n_times);
params=zeros(2,n_times);
%best result comes from initiating with ones and using previous values...
for n=n_start:n_incr:n_max
    n
    index=n/n_incr-(n_start-n_incr)/n_incr;
    if(n~=n_start)
        add=n/n_incr;
        a_mat(1:n,index)=GetA_RelErr( S,K2,t, r,qdiv, sigma,fit_call,n,a_mat(1:n,index-1));%use previous guess
    else
        [a_mat(1:n,index)]=GetA_RelErr( S,K2,t, r,qdiv, sigma,fit_call,n,randn(n_start,1)*2);
    end
    a=a_mat(1:n,index);
    params(:,index)=lsqnonlin(@CostFuncNDF,[r+0.01,sigma],lb,ub);
    call_NDF= Call_NDF(S,K2,params(1,index),a_mat(1:n,index),params(2,index),0,r,t,n);
    mean(abs(call_NDF-fit_call)./max(fit_call,0.01))
end


%% K -fold cross validation
%fit GC params to most folds then test on missing fold
%how will you get the 10 partitions
rng(5);
c=cvpartition(length(K2),'KFold',3);%10
c2=cvpartition(length(K3),'KFold',3);


test_errorNDF=zeros(n_times,1);%test mean relative error
train_errorNDF=zeros(n_times,1);
time_NDF=zeros(n_times,1);

test_errorGC=zeros(5,1);
train_errorGC=zeros(5,1);
time_GC=zeros(6,1);
lb=[-3;0;0;0;-1];%kappa theta sigma v0 rho
ub=[3;2;2;1;1];


%good initial points for lsqnonlin
mu=0.0364;
sigma=0.3;
params=[   2.999943313182165
    0.222530366600774
    1.264974473588708
    0.043621803226818
    -0.381447729993147];

train_errorH_fit=0;
train_errorH_pure=0;

test_errorH_fit=0;
test_errorH_pure=0;

time_H=0;
time_H_Pure=0;
for n=2:2:16
    %     n c.NumTestSets
    for i=1:c.NumTestSets
        %finding a-coefficients in NDF model
        tic
        disp(['Fold ',num2str(i)])
        NDF_call_fit=fit_call(~c.test(i));
        K=K2(~c.test(i));
        seed = RandStream('mt19937ar','Seed',0);
        
        %GetA or GetA_RelErr
        [a]=GetA_RelErr(S,K2(~c.test(i)),t,r,qdiv,sigma,...
            fit_call(~c.test(i)),n,a_mat(1:n,n/2));%training
        %finding mu and sigma in NDF setting
        params=lsqnonlin(@CostFuncNDF,[r+0.01,sigma],[r,0.1],[1,2]);
        mu=params(1);
        sigma=params(2);
        call_NDF=Call_NDF( S,K2,mu,a,sigma,qdiv,r,t,n);
        
        
        train_errorNDF(n/2)=train_errorNDF(n/2)+mean(abs(fit_call(~c.test(i))-call_NDF(~c.test(i)))./max(fit_call(~c.test(i)),0.01))
        test_errorNDF(n/2)=test_errorNDF(n/2)+mean(abs(fit_call(c.test(i))-call_NDF(c.test(i)))./max(fit_call(c.test(i)),0.01))
        time_NDF(n/2)=time_NDF(n/2)+toc;
        
        %for Heston and GC models.
        if n==2
            call_price=fit_call(~c.test(i));%don't need put price
            %fit to raw data
            K=K2(~c.test(i));
            global K;
            global call_price;
            
            tic
            
            params=lsqnonlin(@CostFunc,params,lb,ub);
            
            time_lsqnonlin_fit=toc;
            
            kappa=params(1);theta=params(2);sigma=params(3);v0=params(4);rho=params(5);
            
            call_price=call_price_full(~c2.test(i));%don't need put price
            %fit to spline data
            K=K3(~c2.test(i));
            global K;
            
            %     global K;%could the global part be a problem??
            %     global call_price;
            
            tic
            
            params_pure=lsqnonlin(@CostFunc,params,lb,ub);
            
            time_lsqnonlin_pure=toc;
            kappa_pure=params_pure(1);theta_pure=params_pure(2);sigma_pure=params_pure(3);v0_pure=params_pure(4);rho_pure=params_pure(5);
            
            tic
            %pure is the best
            call_H=Heston_call(params,S,K2,t,0,r);
            time_H1=toc;
            
            tic
            call_H_pure=Heston_call(params_pure,S,K3,t,0,r);
            time_H_P=toc;
            %just error computations
            train_errorH_fit=train_errorH_fit+mean(abs(call_H(~c.test(i))-fit_call(~c.test(i)))./max(fit_call(~c.test(i)),0.01));
            train_errorH_pure=train_errorH_pure+mean(abs(call_H_pure(~c2.test(i))-call_price_full(~c2.test(i)))./max(call_price_full(~c2.test(i)),0.01));
            
            test_errorH_fit=test_errorH_fit+mean(abs(call_H(c.test(i))-fit_call(c.test(i)))./max(fit_call(c.test(i)),0.01));
            test_errorH_pure=test_errorH_pure+mean(abs(call_H_pure(c2.test(i))-call_price_full(c2.test(i)))./max(call_price_full(c2.test(i)),0.01));
            
            time_H=time_H+time_H1+time_lsqnonlin_fit;
            time_H_Pure=time_H_Pure+time_H_P+time_lsqnonlin_pure;
            
            
            [cumulant,time_cumulant]=Cumulant_Sym( v0,x0,kappa,sigma,rho,t,theta,7);
            
            
            %time is the vector of times it took to find each cumulant
            k=log(K2)-r*t;%we've now fit our coefficients, this is why
            %we will be using the whole spline
            
            
            
            for n2=3:6
                tic
                n2
                q=qGenerator( n2, cumulant );%most q values are 0
                x=(k-cumulant(1))/sqrt(cumulant(2));%there is no c0, this is a 101x1 vector
                H=ProbHermitePolynomial(n2+1,x);%n+1 since it starts at
                [J,I]=JIGenerator( k,cumulant,n2+1,H );
                [yin_call, yin_put]=GCHestonCall( cumulant,k,n2,J,I,H,q );
                train_errorGC(n2-2)=train_errorGC(n2-2)+mean(abs(fit_call(~c2.test(i))-yin_call(~c2.test(i)))./...
                    max(fit_call(~c2.test(i)),0.01));
                test_errorGC(n2-2)=test_errorGC(n2-2)+mean(abs(fit_call(c2.test(i))-yin_call(c2.test(i)))./...
                    max(fit_call(c2.test(i)),0.01));
                
                time_GC(n2-2)=time_GC(n2-2)+sum(time_cumulant(1:n2))+time_lsqnonlin+toc;
                
            end
            
        end
    end
end
test_errorNDF=test_errorNDF/c.NumTestSets;%0.0064
test_errorGC=test_errorGC/c2.NumTestSets;%0.1893

% train_errorNDF=train_errorNDF/c.NumTestSets;%0.0064
% train_errorGC=train_errorGC/c2.NumTestSets;%0.1893
%
% time_NDF=time_NDF/c.NumTestSets;%0.0064
% time_GC=time_GC/c2.NumTestSets;%0.1893
%
% train_errorH_fit=train_errorH_fit/c.NumTestSets
% train_errorH_pure=train_errorH_pure/c.NumTestSets
%
% test_errorH_fit=test_errorH_fit/c.NumTestSets
% test_errorH_pure=test_errorH_pure/c.NumTestSets
%fit a params, get mu and sigma on most folds test on missing fold
% hold off
% plot(K2(c.test(c.NumTestSets)),yin_call)
% hold on
% plot(K2(c.test(c.NumTestSets)),fit_call(c.test(10)))
% legend('GC Call','Actual Call')
% xlabel('Strike')
% ylabel('Price')
% %
% %
% plot(K2(c.test(10)),abs(yin_call-fit_call(c.test(10)))./max(fit_call(c.test(10)),0.01))
% xlabel('Strike')
% ylabel('Relative Error')


% hold off
% plot(K2(c.test(10)),call_NDF)
% hold on
% plot(K2(c.test(10)),fit_call(c.test(10)))
%
% legend('GH Call, n=10','Actual Call')
% xlabel('Strike')
% ylabel('Price')
%
% hold off
% plot(K2(c.test(10)),abs(call_NDF-fit_call(c.test(10)))./max(fit_call(c.test(10)),0.01))
% xlabel('Strike')
% ylabel('Relative Error')
