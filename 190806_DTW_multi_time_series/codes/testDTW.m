function test=testDTW(a,a1,b,b1)

% a=randn(300,1);
% a1=randn(300,1);
% b=randn(300,1);
% b1=randn(300,1);
count=0;
for i=1:1
% a=randn(300,1);
% a1=randn(300,1);
% b=randn(300,1);
% b1=randn(300,1);
    if ((dtw(a,a1,15)+ dtw(b,b1,15))<dtw([a b],[a1 b1],15))
        count=count+1;
    end 
end
test=count;
end