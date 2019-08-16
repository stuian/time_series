function [IDX]=DBSCAN(DTW,epsilon,MinPts)  
    C=0;
    n=size(DTW,1);
    IDX=zeros(n,1);  
%     D=pdist2(X,X);  
    visited=false(n,1);  
    isnoise=false(n,1);  
    for i=1:n  
        if ~visited(i)  
            visited(i)=true;  
            Neighbors=RegionQuery(i);  
            if numel(Neighbors)<MinPts  
                % X(i,:) is NOISE  
                isnoise(i)=true;  % ？？为什么在neighbors小于minpits的时候，直接就被设置为了noise点呢？
            else  %若为核心点，则将C（类标号）加一，并且运行类扩展函数。
                C=C+1;  
                ExpandCluster(i,Neighbors,C);  
            end
        end
    end
      
    function ExpandCluster(i,Neighbors,C)  %类扩展函数
        IDX(i)=C;  
        k = 1;  
        while true  %始终为真的循环，依赖break才可以跳出
            j = Neighbors(k);  %j依次为neighbors中的每一个下标
            if ~visited(j)  %如果j还没有处理过，则在这里判断它是否为核心点
                visited(j)=true;  
                Neighbors2=RegionQuery(j);  
                if numel(Neighbors2)>=MinPts  
                    Neighbors=[Neighbors Neighbors2];   %扩展neighbors  
                end
            end  
            %对于邻域里面每一个IDX为0的数据，都将它分到C类中
            if IDX(j)==0
                IDX(j)=C;  
            end 
            k = k + 1;  
            if k > numel(Neighbors)  %neighbor中的下标全部循环结束
                break;  %可以跳出while循环
            end  
        end  
    end  
      
    function Neighbors=RegionQuery(i)  
        Neighbors=find(DTW(i,:)<=epsilon);  
    end
end  

