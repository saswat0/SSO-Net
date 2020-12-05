function [bfit,fbestval,bestsol,time] = GSO(pop,fname,LowerBound,UpperBound,MaxIter)

% function [fbestval,bestparticle] = GSO(fname,NDim,MaxIter)
%
%   Run a Group Search Optimizer algorithm
%
% Input Arguments:
%   fname       - the name of the evaluation .m function
%   NDim        - dimension of the evalation function
%   MaxIter     - maximum iteration


% *******************************************
% Group Search Optimizer algorithm for Matlab
% *******************************************
% Copyright (C) 2004-2008 Shan He, Q. H. Wu and J. R. Sanders
% The University of Liverpool
% Intelligence Engineering & Automation Group
%
% Last modifed 26-August-08
[PopSize,NDim] = size(pop);
t0  = clock;
flag = 0;
iteration = 0;

%MaxIter =1000;  % maximum number of iteration
% PopSize = 48;     % population of particles

initial_angle = pi/4.*ones(NDim-1,PopSize);
angle = initial_angle;
leftangle = angle;
rightangle = angle;

Bound=feval(fname,pop);
% Defined lower bound and upper bound.
LowerBound = LowerBound';
UpperBound = UpperBound';
% for i=1:PopSize
%     LowerBound(:,i) = Bound(:,1);
%     UpperBound(:,i) = Bound(:,2);
% end

DResult = 1e-1;    % Desired results
population = pop';     % Initialize GSO population

vmax = ones(NDim,PopSize);
for i = 1:NDim
    vmax(i,:) = (UpperBound(i,:)-LowerBound(i,:));
end
l_max = norm(vmax(:,1));
distance = l_max*repmat(ones(1,PopSize),NDim,1);

% Constant a
a = round(((NDim+1)^.5));

max_pursuit_angle = (pi/(a^2));
max_turning_angle = max_pursuit_angle/2;

% Calculate initial directions from initial angles
for j = 1:PopSize
    direction(1,j) = (cos(angle(1,j)));
    for i=2:NDim-1
        direction(i,j) = cos(angle(i,j)).*prod(sin(angle(i:NDim-1,j)));
    end
    direction(NDim,j) = prod(sin(angle(1:NDim-1,j)));
end

% Evaluate initial population
for i = 1:PopSize,
    fvalue(i) = feval(fname,population(:,i)');
end

% Prevent memeber from flying outside search space
OutFlag = population<=LowerBound | population>=UpperBound;
population = population - OutFlag.*distance.*direction;

% Finding best memeber in initial population
[fbestval,index] = min(fvalue);    % Find the globe best
bestmember = population(:,index);

oldangle = angle;
oldindex = index;
badcounter = 0;



while(flag == 0) & (iteration < MaxIter)

    iteration = iteration +1;

    for j = 1:PopSize
        R1 = randn(1);
        R2 = rand(NDim-1,1);
        R3 = rand(NDim, 1);

        
        if j == index % Select the best member as producer, it stops and searches.
            distance(:,j)=l_max*R1;
            
            SamplePosition = [];
            SampleAngle = [];
            SampleValue = [];
            SampleDirection = [];
          
            % If the producer cannot find a better area after $a$ iterations,
            % it will turn its head back to zero degree
            if badcounter>=a
                angle(:,j)= oldangle(:,j);
            end

            % Save current fitness value to SamplePosition
            SamplePosition = [SamplePosition,population(:,j)];
            SampleAngle = [SampleAngle,angle(:,j)];
            SampleValue = [SampleValue,fvalue(j)];
            SampleDirection = [SampleDirection, direction(:,j)];
            
            % Look Straight
            direction(1,j)=prod(cos(angle(1:NDim-1,j)));
            for i=2:NDim-1
                direction(i,j)=sin(angle(i,j)).*prod(cos(angle(i:NDim-1,j)));
            end
            direction(NDim,j)=sin(angle(NDim-1,j));
            
            StraightPosition = population(:,j) + distance(:,j).*direction(:,j);   
            Outflag = (StraightPosition>UpperBound(:,j) | StraightPosition<LowerBound(:,j));
            StraightPosition = StraightPosition - Outflag.*distance(:,j).*direction(:,j);
%             Straightfunction = strcat(fname,StraightPosition');
            Straightfvalue = feval(fname,StraightPosition');
            SamplePosition = [SamplePosition,StraightPosition];
            SampleAngle = [SampleAngle,angle(:,j)];
            SampleValue = [SampleValue,Straightfvalue];
            SampleDirection = [SampleDirection, direction(:,j)];

            % Look left
            leftangle = angle(:,j) + max_pursuit_angle.*R2/2;  %look left
            direction(1,j)=prod(cos(leftangle(1:NDim-1)));
            for i=2:NDim-1
                direction(i,j)=sin(leftangle(i)).*prod(cos(leftangle(i:NDim-1)));
            end
            direction(NDim,j)=sin(leftangle(NDim-1));  
            
            LeftPosition = population(:,j) + distance(:,j).*direction(:,j); 
            Outflag = (LeftPosition>UpperBound(:,j) | LeftPosition<LowerBound(:,j));
            LeftPosition = LeftPosition-Outflag.*distance(:,j).*direction(:,j);
            % Save the value to a vector SamplePosition
            Leftfvalue = feval(fname,LeftPosition');
            SamplePosition = [SamplePosition,LeftPosition];
            SampleAngle = [SampleAngle,leftangle(:)];
            SampleValue = [SampleValue,Leftfvalue];
            SampleDirection = [SampleDirection, direction(:,j)];

            % Look right
            rightangle = angle(:,j) - max_pursuit_angle.*R2/2; %look right 
            direction(1,j)=prod(cos(rightangle(1:NDim-1)));
            for i=2:NDim-1
                direction(i,j)=sin(rightangle(i)).*prod(cos(rightangle(i:NDim-1)));
            end
            direction(NDim,j)=sin(rightangle(NDim-1));   
            
            RightPosition = population(:,j) + distance(:,j).*direction(:,j);
            Outflag = (RightPosition>UpperBound(:,j) | RightPosition<LowerBound(:,j));
            RightPosition = RightPosition - Outflag.*distance(:,j).*direction(:,j);
            % Save the value to a vector SamplePosition
            Rightfvalue = feval(fname,RightPosition');
            SamplePosition = [SamplePosition,RightPosition];
            SampleAngle = [SampleAngle,rightangle(:)];
            SampleValue = [SampleValue,Rightfvalue];
            SampleDirection = [SampleDirection, direction(:,j)];
            
            % sample 4 points and fly/stay to/at the best position
            [fbestdirctionval, best_position_ID] = min(SampleValue);
            population(:,j) = SamplePosition(:,best_position_ID);


            if best_position_ID ~= 1   % if the producer finds a better place (1 means current position)
                angle(:,j) = SampleAngle(:,best_position_ID);
                oldangle(:,j) = angle(:,j);
                badcounter = 0;
            else                    % if the producer stays
                badcounter = badcounter+1;
                angle(:,j) =  angle(:,j) + max_turning_angle.*R2; % Turn and sample a new direction
            end


        else

            angle(1:NDim-1,j) = angle(1:NDim-1,j) + max_turning_angle.*R2;           

            if rand(1)>.2
                % Scroungers' behaviour
                distance(:,j) = R3.*(bestmember-population(:,j));
                population(:,j) = population(:,j) + distance(:,j);
            else
                % Rangers' behaviour 
                distance(:,j)=l_max*repmat(a*R1,NDim,1);
                % direction calculation
                direction(1,j)=(cos(angle(1,j)));
                for i=2:NDim-1
                    direction(i,j)=cos(angle(i,j)).*prod(sin(angle(i:NDim-1,j)));
                end
                direction(NDim,j)=prod(sin(angle(1:NDim-1,j)));
                population(:,j) = population(:,j) + distance(:,j).*direction(:,j);
            end
        end
        

    end

    % Prevent members except producer from flying outside search space
    OutFlag = population<=LowerBound | population>=UpperBound;
    population = population - OutFlag.*distance.*direction;

    % Evaluate the new group members
    for i = 1:PopSize
        fvalue(i) = feval(fname,population(:,i)');
    end

    % Updating index
    [fbestval(iteration), index] = min(fvalue);

    bestmember=population(:,index);
    

%     if iteration/10==floor(iteration/10)
%         %fprintf(1,'\n');
%         fprintf(1,'%e   ',fbestval);
%         fprintf(1,'\n');
%         Best(iteration/10) =fbestval;
%         if fbestval == 0
%             Best((iteration/10):300) = 0;
%             break
%         end        
%     end



end
bfit = fbestval(end);
bestsol = bestmember';
time = toc;

