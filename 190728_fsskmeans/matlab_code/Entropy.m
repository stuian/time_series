%entropy ¼´Îª¡°ìØ¡±
function epy=Entropy(idx,labels)

en=zeros(1,max(idx)-min(idx)+1);
for k1=min(idx):max(idx)
    ind=find(idx==k1);
    temp=0;
    for k2=min(labels):max(labels)
        mm=find(labels(ind)==k2);
        if(~isempty(mm))
            p=length(mm)/length(ind);
            temp=temp+p*log2(p);
        end
    end
    en(k1)=-temp;
end
ind=[];
for k1=min(idx):max(idx)
    ind(k1)=length(find(idx==k1));
end 
epy=sum(ind.*en/length(idx));
end