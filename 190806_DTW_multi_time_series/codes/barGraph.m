function myBar = barGraph()

I=load('classesI.txt');
D=load('classesD.txt');
I(end+1,:)=sum(I);
D(end+1,:)=sum(D);
I=I(:,1)./I(:,2);
D=D(:,1)./D(:,2);
I=1-I;
D=1-D;
bar([D I]);

end