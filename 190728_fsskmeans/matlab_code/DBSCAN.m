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
                isnoise(i)=true;  % ����Ϊʲô��neighborsС��minpits��ʱ��ֱ�Ӿͱ�����Ϊ��noise���أ�
            else  %��Ϊ���ĵ㣬��C�����ţ���һ��������������չ������
                C=C+1;  
                ExpandCluster(i,Neighbors,C);  
            end
        end
    end
      
    function ExpandCluster(i,Neighbors,C)  %����չ����
        IDX(i)=C;  
        k = 1;  
        while true  %ʼ��Ϊ���ѭ��������break�ſ�������
            j = Neighbors(k);  %j����Ϊneighbors�е�ÿһ���±�
            if ~visited(j)  %���j��û�д���������������ж����Ƿ�Ϊ���ĵ�
                visited(j)=true;  
                Neighbors2=RegionQuery(j);  
                if numel(Neighbors2)>=MinPts  
                    Neighbors=[Neighbors Neighbors2];   %��չneighbors  
                end
            end  
            %������������ÿһ��IDXΪ0�����ݣ��������ֵ�C����
            if IDX(j)==0
                IDX(j)=C;  
            end 
            k = k + 1;  
            if k > numel(Neighbors)  %neighbor�е��±�ȫ��ѭ������
                break;  %��������whileѭ��
            end  
        end  
    end  
      
    function Neighbors=RegionQuery(i)  
        Neighbors=find(DTW(i,:)<=epsilon);  
    end
end  

