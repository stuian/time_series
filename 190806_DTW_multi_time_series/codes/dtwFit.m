function fit = dtwFit(s,t,path)
fit=[];
i=0;
j=0;

for p=1:length(path)
  if(i<=size(s,1) && j<=size(t,1))  
   if (path(p)==1) 
     i=i+1;
     fit=[fit (norm(s(i,:)-t(j,:)))^2] ;

   elseif (path(p)==2)
      j=j+1;
      fit=[fit (norm(s(i,:)-t(j,:)))^2] ;
      
   elseif(path(p)==3)
       i=i+1;
       j=j+1;
       fit=[fit (norm(s(i,:)-t(j,:)))^2] ;

   end
  end
end



end