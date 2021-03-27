%%  Two-state Markov chain model of coin
%
%   For pht=pth the coin is fair but subsequent flips are correlated
%
clear; clc; close all;
 
% probability of heads followed by tails and vice-versa
pht = 0.1; pth = 0.3;
 
% transition matrix (rows sum to 1)
phh = 1-pht; ptt = 1-pth;  P = [ phh pht ; pth ptt ];
 
xh0 = 1; % initial probability of being heads
xt0 = 1-xh0; % xh0+xt0=1
 
%% Calculate and plot discrete time map x_{k+1} = x_{k}*P
nflips = 50;
x = zeros(nflips,2);
x(1,:) = [ xh0  xt0 ]; % initial probabilities (row vector)
for k=2:nflips;
    x(k,:) = x(k-1,:)*P;
end
xfinal = x(end,:);
% disp(['Norm of change from last iteration: ' num2str(norm(xfinal-xfinal*P))])
% figure
% subplot(2,1,1)
% plot(1:nflips,x(:,1),'r-o','MarkerFaceColor','r'); hold on
% plot(1:nflips,x(:,2),'b-o','MarkerFaceColor','b');
% xlabel('discrete time')
% ylabel('prob')
 
%% The steady-state probability distribution
%
%  xss may be found more directly as the as Perron vector of P,
%  which is the eigenvector of P with eigenvalue 1.
%
%  See https://en.wikipedia.org/wiki/Perron–Frobenius_theorem
%
%  xss calculated in this way should agree with limiting value of x
%  calculated through iteration, provided enough flips occur (k large).
%
[V,D]=eig(P'); %
d=diag(D); a=find(d==1); % index 1 or 2 corresonds to eigenvalue 1
xss = V(:,a)/sum(V(:,a)); % normalize Perron vector
% plot(nflips,xss(1),'r*',nflips,xss(2),'b*','MarkerSize',30);
% legend({'Pr[X_i=H]','Pr[X_i=T]','steady-state','steady-state'})
 
%% Calculate and plot instance of a sequence of flips
X=zeros(nflips,1);
if rand<xh0, X(1)=1; else X(1)=0; end
for k=2:nflips;
    if X(k-1)==1 % heads
        if rand<pht, X(k)=0; end
    else % tails, X(i-1)=0
        if rand<pth, X(k)=1; end
    end
end
% subplot(2,1,2)
% stairs(1:nflips,X,'k-o','MarkerFaceColor','k');
% xlabel('discrete time')
% ylabel('state')
% axis([-Inf Inf -0.5 1.5])
%  
 
%% Calcuate and plot N instances of the Markov chain
%
%  Notice that the code is "vectorized."
%  There is a k=2..nflips outer loop,
%  but no 1..N inner loop
%
N=100000;
Y=NaN(nflips,N);
Y(1,:)=(rand(1,N)<xh0); % initial states of N coins
for k=2:nflips;
    heads = find(Y(k-1,:)==1); Y(k,heads)=1;
    tails = find(Y(k-1,:)==0); Y(k,tails)=0;
    heads_to_tails = find(rand(size(heads))<pht);
    Y(k,heads(heads_to_tails)) = 0;
    tails_to_heads = find(rand(size(tails))<pth);
    Y(k,tails(tails_to_heads)) = 1;
end
% figure
% Yoffset = 2*ones(nflips,1)*(0:N-1);
% stairs(1:nflips,Y+Yoffset);
% axis([-Inf Inf -0.5 2*N-0.5])
% xlabel('discrete time')
% ylabel('state')
% set(gca,'YTickLabel',[]);
%  
%% Average N instances and compare to dynamics of x_{k}
figure
Yavg = sum(Y,2)/N;
plot(1:nflips,x(:,1),'ro-','MarkerFaceColor','r'); hold on;
plot(1:nflips,Yavg,'ro--');
plot(1:nflips,x(:,2),'bo-','MarkerFaceColor','b')
plot(1:nflips,1-Yavg,'bo--')
axis([-Inf Inf 0 1])
xlabel('discrete time')
ylabel('prob')
legend({'Pr[X_i=H]','observed','Pr[X_i=T]','observed'})
 
%% Calculate correlation of X_{k+1} and X_{k}
 
funny_coin_corr = corr(X(1:end-1),X(2:end));
disp(['The correlatino of X_{k+1} and X_{k} is ' num2str(funny_coin_corr ) '.'])
 
%% Calculate and plot the correlation function
%
%  c(lag) = average of X_{k}*X_{k-lag} over all valid k
%
%  The plot is the average c(lag) over N trials
%
% figure
% for n=1:N
%     funny_coin_autocorr_fn(:,n) = autocorr(Y(:,n));
% end
% plot(1:size(funny_coin_autocorr_fn,1),mean(funny_coin_autocorr_fn,2),'bo-')
% xlabel('discrete time lag')
% ylabel('autocorrelation')
% title('c(lag) = average over k of X_{k}*X_{k-lag}')
% axis([-Inf Inf -Inf Inf])
% grid on