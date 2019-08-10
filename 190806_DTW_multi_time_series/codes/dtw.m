% Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>,
% Signal Analysis and Machine Perception Laboratory,
% Department of Electrical, Computer, and Systems Engineering,
% Rensselaer Polytechnic Institute, Troy, NY 12180, USA

% dynamic time warping of two signals

function [d dist path]=dtw(s,t,w)
% s: signal 1, size is ns*k, row for time, colume for channel 
% t: signal 2, size is nt*k, row for time, colume for channel 
% w: window parameter
%      if s(i) is matched with t(j) then |i-j|<=w
% d: resulting distance

if nargin<3
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
D(1,1)=0;

%% begin dynamic programming
for i=1:ns
    for j=max(i-w,1):min(i+w,nt)
        oost=norm(s(i,:)-t(j,:));
        oost=(oost)^2;
        D(i+1,j+1)=oost+min( [D(i,j+1), D(i+1,j), D(i,j)] );
        
    end
end
d=sqrt(D(ns+1,nt+1));

i=ns;
j=nt;
dist=[];
path=[];
while (i~=0 || j~=0)
 if (min( [D(i,j+1), D(i+1,j), D(i,j)] )==D(i,j+1))
     dist=[(norm(s(i,:)-t(j,:)))^2 dist];
     path=[1 path];
     i=i-1;
 elseif (min( [D(i,j+1), D(i+1,j), D(i,j)] )==D(i+1,j))
     dist=[(norm(s(i,:)-t(j,:)))^2 dist];
     path=[2 path];
     j=j-1;
 elseif (min( [D(i,j+1), D(i+1,j), D(i,j)] )==D(i,j))
     dist=[(norm(s(i,:)-t(j,:)))^2 dist];
     path=[3 path];
     i=i-1;
     j=j-1;
 end
    
end

end

