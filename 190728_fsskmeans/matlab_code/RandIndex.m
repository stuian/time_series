function RI=RandIndex(train_label,labels)
len=length(labels);
TP = 0;
TN = 0;
FP = 0;
FN = 0;
for k1=1:len-1
    for k2=k1+1:len
        if (train_label(k1)==train_label(k2)&&labels(k1)==labels(k2))
            TP = TP+1;
        elseif (train_label(k1)~=train_label(k2)&&labels(k1)~=labels(k2))
            TN = TN+1;
        elseif (train_label(k1)==train_label(k2)&&labels(k1)~=labels(k2))
            FP = FP+1;
        elseif (train_label(k1)~=train_label(k2)&&labels(k1)==labels(k2))
            FN = FN+1;
        end
    end
end
RI = (TP+TN)/(TP+FP+FN+TN);
end
% function RI=RandIndex(indx,train_label)
% len=length(train_label);
% total=0;
% for k1=1:len-1
%     for k2=k1+1:len
%         if(indx(k1)==indx(k2)&&train_label(k1)==train_label(k2))
%             total=total+1;
%         elseif(indx(k1)~=indx(k2)&&train_label(k1)~=train_label(k2))
%             total=total+1;
%         end
%     end
% end
% RI=total*2/len/(len-1);
% end