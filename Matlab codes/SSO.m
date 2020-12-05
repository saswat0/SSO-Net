function[bestfit,fbst,best_sub,time] = SSO(val,objfun,x_min,x_max,kmax)
[N,D] = size(val);
%Initialization of SSO parameters
alphak = rand();
betak = rand();
t = 1;       % time interval stage of k
gk = rand();
x = val;
f = feval(objfun,x);
% find global best and particle best
[fgbest,igbest]=min(f);
gbest = x(igbest,:);
pbest=x; fpbest = f;

tic;
for k = 1:kmax
    for i = 1:N
        for j = 1:D
            r1 = rand();
            delf = abs(fgbest-f(i))/fgbest;
            term1 = gk*r1*delf;
            if k == 1
                vel(i,j) = term1;
            else
                term2 = min([abs(term1)+alphak*r1*vel(i,j) abs(betak*vel(i,j))]);
                vel(i,j) = term2;
            end
            
            
        end
        if rand<0.5
            x(i,:) = x(i,:)+vel(i,:)*t;         % forward movement update process by Eq. (10.9)
        else
            r3 = 2*rand-1;                       % (b-a)*rand(1,100)+a (for random values )
            x(i,:) = x(i,:)+(r3.*x(i,:));          % rotation movement update process by Eq. (10.10) 
        end
    end
    % bound check
    for mi=1:N
        for mj=1:D
            if x(mi,mj)<x_min(mi,mj)
               x(mi,mj) = x_min(mi,mj);
            else
                if x(mi,mj)>x_max(mi,mj)
                    x(mi,mj)=x_max(mi,mj);
                end
            end
        end
    end
    f = feval(objfun,x);
    [minf,iminf]=min(f);
    if minf<= fgbest
        fgbest = minf; gbest = x(iminf,:);
        best_sub(k,:) = x(iminf,:);
        fbst(k) = minf;
    else
        fbst(k) = fgbest;
        best_sub(k,:) = gbest;
    end
    inewpb = find(f<=fpbest);
    pbest(inewpb,:) = x(inewpb,:);
    fpbest(inewpb) = f(inewpb);
end
time = toc;
bestfit = fbst(end);
end