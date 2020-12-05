function[bestfit,fbst,best_sub,time] = pso(val,objfun,L,U)
[N,D] = size(val);
% Initialization of PSO parameters
c1=2; c2 = 2;         % Acceleration constants
wmax = 0.9; wmin = 0.1;  % maximum and minimum weight assignment
x_max = U; x_min =L; %maximum and minimum values of solution
w= zeros(1,itermax);       % Initialization of weight
for iter=1:itermax
    w(iter)= wmax-((wmax-wmin)/itermax)*iter;   % Inertia weight update
end
m = L;
n = U;
q=(n-m)/(D*2);  % Initial velocity
Ki = 1;

% Random initialization of position and velocity
x = val;
v = q.*rand(N,D);
f = feval(objfun,x); % Evaluate objective for all particles
% find global best and particle best
[fgbest,igbest]=min(f);  
gbest = x(igbest,:);     
pbest=x; fpbest = f;     
tic; cnt = 0; it = 1;
%% Iterate
while cnt>= 5
    % Update velocities and position
    v(1:N,1:D) = w(it)*v(1:N,1:D)+c1*rand*(pbest(1:N,1:D)-x(1:N,1:D))+c2*rand*(repmat(gbest,N,1)-x(1:N,1:D));
    x(1:N,1:D)= x(1:N,1:D)+v(1:N,1:D);
    for mi=1:N
        for mj=1:D
            if x(mi,mj)<x_min(mi,mj)
               x(mi,mj) = x_min(mi,mj);
            else
                if x(mi,mj)>x_max(mi,mj)
                    x(mi,mj)=x_max(mi,mj);
                else
                end
            end
        end
    end
    f = feval(objfun,x);  % Evaluate objective for all particles
    % Find global best and Particle best
    [minf,iminf]=min(f);
    if minf<= fgbest
       fgbest = minf; gbest = x(iminf,:);
       best_sub = x(iminf,:);
       fbst(it) = minf;
    else
       fbst(it) = fgbest;
       best_sub = gbest;
    end
    inewpb = find(f<=fpbest);
    pbest(inewpb,:) = x(inewpb,:);
    fpbest(inewpb) = f(inewpb);
    
    if it>=2
        if fbst(it-1) == fbst(it)
            cnt = cnt+1;
        else
            cnt = 0;
        end
    end
    it = it+1;
end
Ki = Ki+1;
time = toc;
bestfit = fbst(end);
end