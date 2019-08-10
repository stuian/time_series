function [d d1 d11 d22]=dtwI(s,t,s1,t1,w)


if nargin<5
    w=Inf;
end

ns=length(s);
nt=length(t);
if size(s,2)~=size(t,2)
    error('Error in dtw(): the dimensions of the two input signals do not match.');
end
w=max(w, abs(ns-nt)); % adapt window size

%% initialization
D=zeros(ns+1,nt+1)+Inf; % cache matrix
path=zeros(ns,nt); %path matrix
D(1,1)=0;

%% begin dynamic programming
for i=1:ns
    for j=max(i-w,1):min(i+w,nt)
        oost=norm(s(i,:)-t(j,:));
        D(i+1,j+1)=oost+min( [D(i,j+1), D(i+1,j), D(i,j)] );
        if ( (min( [D(i,j+1), D(i+1,j), D(i,j)] ))==D(i,j+1) )
            path(i,j)=1;
        elseif ( (min( [D(i,j+1), D(i+1,j), D(i,j)] ))==D(i+1,j) )
                path(i,j)=2;
        elseif ( (min( [D(i,j+1), D(i+1,j), D(i,j)] ))==D(i,j) )
            path(i,j)=3;
        end
    end
end
d=D(ns+1,nt+1);

%% calculate the second pair
ns=length(s1);
nt=length(t1);
if size(s1,2)~=size(t1,2)
    error('Error in dtw(): the dimensions of the two input signals do not match.');
end
w=max(w, abs(ns-nt)); % adapt window size

% initialization
D=zeros(ns+1,nt+1)+Inf; % cache matrix
path1=zeros(ns,nt); %path matrix
D(1,1)=0;

% begin dynamic programming
for i=1:ns
    for j=max(i-w,1):min(i+w,nt)
        oost=norm(s1(i,:)-t1(j,:));
        D(i+1,j+1)=oost+min( [D(i,j+1), D(i+1,j), D(i,j)] );
        if ( (min( [D(i,j+1), D(i+1,j), D(i,j)] ))==D(i,j+1) )
            path1(i,j)=1;
        elseif ( (min( [D(i,j+1), D(i+1,j), D(i,j)] ))==D(i+1,j) )
                path1(i,j)=2;
        elseif ( (min( [D(i,j+1), D(i+1,j), D(i,j)] ))==D(i,j) )
            path1(i,j)=3;
        end
    end
end
d1=D(ns+1,nt+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calculate the first pair with path1
ns=length(s);
nt=length(t);
if size(s,2)~=size(t,2)
    error('Error in dtw(): the dimensions of the two input signals do not match.');
end
w=max(w, abs(ns-nt)); % adapt window size

% initialization
D=zeros(ns+1,nt+1)+Inf; % cache matrix
D(1,1)=0;

% begin dynamic programming
for i=1:ns
    for j=max(i-w,1):min(i+w,nt)
        oost=norm(s(i,:)-t(j,:));
        if ( path1(i,j)==1 )
            D(i+1,j+1)=oost+D(i,j+1);
        elseif ( path1(i,j)==2 )
                D(i+1,j+1)=oost+D(i+1,j);
        elseif ( path1(i,j)==3 )
                D(i+1,j+1)=oost+D(i,j);
        else
            error('path1 is missing the path');
        end
        
    end
end
d11=D(ns+1,nt+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calculate the second pair with path
ns=length(s1);
nt=length(t1);
if size(s1,2)~=size(t1,2)
    error('Error in dtw(): the dimensions of the two input signals do not match.');
end
w=max(w, abs(ns-nt)); % adapt window size

% initialization
D=zeros(ns+1,nt+1)+Inf; % cache matrix
D(1,1)=0;

% begin dynamic programming
for i=1:ns
    for j=max(i-w,1):min(i+w,nt)
        oost=norm(s1(i,:)-t1(j,:));
        if ( path(i,j)==1 )
            D(i+1,j+1)=oost+D(i,j+1);
        elseif ( path(i,j)==2 )
                D(i+1,j+1)=oost+D(i+1,j);
        elseif ( path(i,j)==3 )
                D(i+1,j+1)=oost+D(i,j);
        else
            error('path is missing the path');
        end
        
    end
end
d22=D(ns+1,nt+1);
end
