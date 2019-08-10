function distance = dtwVisual(query,can_seq,force_plots)






global matrix;	 % Define a global matrix, which will be accessible to the local functions.
global M;	 % Uppercase M is a constant global accessible to the local functions.
   M = length(query);	 % Find out how long the query is.

global N;	 % Uppercase N is a constant global accessible to the local functions.
       N = length(can_seq);	    % Find out how long the candidate sequence is.


global WARPING_WINDOW ;	 % This variables value is the max number of cells, away from the diagonal, that the search
WARPING_WINDOW =85;     % algorithm can visit. A low value speeds up search and constains the amount of warping,
% but may miss a better solution. A 'good' value is ->  WARPING_WINDOW = ceil(min(M,N)/2);
                              
                              
                              
                              
                                                          
                             
                              
                            
                

matrix   = (query * ones(1,N) - (can_seq * ones(1,M))').^2;	 % Build the matrix to be searched.


distance = sqrt(d(M,N));	 % Search the matrix.
path     = extract_path(matrix);	 % Recover the warping path.

if nargin == 3 plot_warping(query,can_seq,path);  end; % Spawn plots if user requested them.




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Local Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function dist = d(m,n)	             % Recurisive function: Inputs are the row and column considered
% Output is overall distance.

global matrix;	 % Declare the same global matrix as above.
global M;	 % Declare the matrix row size (the full matrix).
global N;	 % Declare the matrix column size (the full matrix).
global WARPING_WINDOW;	     % Declare the size of warping window.



if abs(n-(N/(M/m))) > WARPING_WINDOW	 % Test to see if this point is outside the warping window.
  % It IS outside the warping window.
   if n - (N /  (M/m)) > 0	 % Is it ABOVE the warping window?
      matrix(1:m,n:N) = nan;	 % Every point above and to the right is set to NAN.
   else	 % Is it BELOW the warping window?
      matrix(m:M,1:n) = nan;	 % Every point below and to the left is set to NAN.
   end;
end;



if matrix(m,n) < 0	| isnan(matrix(m,n))	% If you find a NEGATIVE value that means that the best possible path to this
   dist = abs(matrix(m,n));	 % location has already been calculated. The cost to this point is the absolute
   return;	 % value of the negative value. So RETURN the absolute value, and DONT recurse.
end;


if(m == 1) & (n == 1) % Case 1: A warpping path has reach the end.	
   dist = matrix(1,1);     % Return the contribution of distance.
   matrix(1,1) = - (matrix(1,1));	        % Negate the value, to record the fact that this cell has been visited.
   return; % RECURSIVE BASE CASE, RETURN
         
elseif(m == 1)	 % Case 2: We are somewhere in the top row of matrix.
   
   [contrib_dist] = d(m,n-1);	     % Only need to consider moving left.
   
   dist = (matrix(m,n))+contrib_dist;       % Return the contribution of distance.
   matrix(m,n) = - dist;	     % Negate the value, to record the fact that this cell has been visited.
   
elseif(n == 1)	 % Case 3: We are somewhere in the left column of matrix.
   
   [contrib_dist] = d(m-1,n);	        % Only need to consider moving left.

   dist = (matrix(m,n))+contrib_dist;	    % Return the contribution of distance.
   matrix(m,n) = - dist; % Negate the value, to record the fact that this cell has been visited.
   
else	 % Case 4: We are somewhere away from the edges of matrix
   
   [contrib_dist1] = d(m-1,n-1);	   % Consider moving diagonally.
   [contrib_dist2] = d(m-1,n);	        % Consider moving left.
   [contrib_dist3] = d(m,n-1);	       % Consider moving up.
   
      % Find out which of the above was the best move.
   
   [value,location] = min([contrib_dist1, contrib_dist2, contrib_dist3]);
   
   if location == 1	 % Diagonal was best.
      dist =   (matrix(m,n)) + value;	% Return the contribution of distance.

   elseif location == 2	 % Left was best.
      dist =   (matrix(m,n)) + value;	% Return the contribution of distance.
   
   else % up was best.
      dist =   (matrix(m,n)) + value;	% Return the contribution of distance.

   end;
      
   matrix(m,n) = - dist;	 % Negate the value, to record the fact that this cell has been visited.
end;	

%End function dist = d(m,n)


%-------------------------------------------------------------------------------------------------------------------

function path     = extract_path(matrix)	     % This function is designed to search the modified matrix for
% the warping path in O(m+n) time. It should be invoked after
                                                            % a call to d(m,n) only.
                                                   
matrix = abs(matrix);	 % Change the values to positive.
path     = size(matrix);	 % The first (working backwards) cell in the path is the bottom right corner.
current  = size(matrix);	 % Initialize the current cell to the above.


while (current(1)> 1 | current(2) > 1)	     % While the current cell is not the top right cell.
   
   
   if  (current(1)> 1 & current(2) > 1)	         % General case: We are in an internal cell.
     dist1 = matrix(current(1)-1,current(2)-1);	        % Consider moving diagonally.
   dist2 = matrix(current(1)-1,current(2)  );	            % Consider moving up.
   dist3 = matrix(current(1)  ,current(2)-1);	            % Consider moving left.
      
   elseif current(1)> 1	 % Special case 1 : We are on the top row.
      dist1 = realmax;	 % DONT consider moving diagonally.
   dist2 = matrix(current(1)-1,current(2)  );              % Consider moving up.
   dist3 = realmax;	 % DONT consider moving left.

   else	 % Special case 2 : We are in the left column.
      dist1 = realmax;	           % DONT consider moving diagonally.
   dist2 = realmax;	            % DONT consider moving left.
   dist3 = matrix(current(1)  ,current(2)-1);	            % Consider moving left.

   end;
   
  [value,location] = min([dist1, dist2, dist3]);          % Find out which of the above was the best move.
   
   if location == 1	        % Diagonal was best.
      current = current  - [1 1];
   elseif location == 2	        % Left was best.
       current = current - [1 0];   
   else    % Up was best.
       current = current - [0 1]; 
   end;

path = [path; current];	 % Append the previous move to the warping path.
     
end;

path = flipud(path);	 % The path was from right to left, so flip it before returning.

%end function path   = extract_path(matrix)





%-------------------------------------------------------------------------------------------------------------------

function plot_warping(query,can_seq,path)	             % This functions spawns 4 different plots which help to visualize the
% algorithms above.


global matrix;	     % Declare the same global matrix as above.
matrix = abs(matrix);	         % Take absolute value of matrix.	
matrix = matrix ./ max(max(matrix));	             % Normalize max value to one.

matrix(:,size(matrix,2)+1) = matrix(:,size(matrix,2));
matrix(size(matrix,1)+1,:) = matrix(size(matrix,1),:);





surf(abs(matrix))	     % Spawn 3d surface.
rotate3d	 % Allow 3d rotation.
hold on;	 % Put hold on for additional elements.


can_seq = can_seq - min(can_seq);	             % Force both sequences to have min values of zero and
can_seq = can_seq / max(can_seq);	             % max values of one, (just for the sake of plotting).
query   = query   - min(query);
query   = query   / max(query);








plot3([1:length(can_seq)],zeros(size(can_seq))+length(query),can_seq + 1,'b'); % Plot the candiate sequence (in 3d)
plot3( zeros(size(query))+length(can_seq) ,[1:length(query)],query + 1,'r');   % Plot the query sequence (in 3d)
   
dx=1;
dy=1;
   
for i = 1 : size(path,1)         % plot a set of patches which map out the warping path.
   
  y = max([matrix(path(i,1),path(i,2)) matrix(path(i,1),path(i,2)) matrix(path(i,1),path(i,2)) matrix(path(i,1),path(i,2))  ]) ;
  patch([(path(i,2)) (path(i,2)) (path(i,2)+ dx) (path(i,2) +dx)  ],[path(i,1) path(i,1)+dy  path(i,1)+dy path(i,1)   ],[y y y y ]+0.05,'r');
      
end;

axis off;

figure;
hold on;
pcolor(abs(matrix));	


factor = min([length(query) length(can_seq)])/3;

plot([1:length(can_seq)],(can_seq * factor)+ 1 + length(query) ,'b'); % Plot the candiate sequence 
plot(( abs(query-1) * factor) + 1 + length(can_seq) ,[1:length(query)],'r');       % Plot the query sequence 

for i = 1 : size(path,1)                                                            % plot a set of patches which map out the warping path.

  y = max([matrix(path(i,1),path(i,2)) matrix(path(i,1),path(i,2)) matrix(path(i,1),path(i,2)) matrix(path(i,1),path(i,2))  ]) ;
  patch([(path(i,2)) (path(i,2)) (path(i,2)+ dx) (path(i,2) +dx)  ],[(path(i,1)-0) (path(i,1)+dy)  (path(i,1)+dy) (path(i,1)-0)   ],[y y y y ]+0.25,'r');
      
end;
  
axis off;




    
%   figure;       % NEW figure
%   hold on;	 % Plot the query and canidate sequence on top of each other.
%   plot(can_seq,'b')
%   plot(query,'r');
%   axis off;
  
  
  figure;	% This is the aligment plot

  hold on;
  axis off;
  
  offset =  2.3; 
  align_adjust =  floor(length(query)/2) - floor(length(can_seq)/2); 
  align_adjust = 0; % TEMP
    
  plot([1:length(can_seq)] + align_adjust, can_seq,'b')
  plot(query+offset,'r'); 
  
     
  for i = 1 :4: size(path,1)     
   
      line([ path(i,1) path(i,2)+ align_adjust ],[ query(path(i,1))+ offset can_seq(path(i,2)) ],'linewidth',0.1,'color',[.5 .5 .5])
  end;