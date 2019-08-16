function f=F_Measure(idx,label)
tp=0;
fp=0;
tn=0;
fn=0;


for k1=1:length(label)-1
    for k2=k1+1:length(label)
        if(label(k1)==label(k2)&&idx(k1)==idx(k2))
            tp=tp+1;
        else 
            if(label(k1)~=label(k2)&&idx(k1)==idx(k2))
                fp=fp+1;
            else
                if(label(k1)==label(k2)&&idx(k1)~=idx(k2))
                    fn=fn+1;
                else
                    if(label(k1)~=label(k2)&&idx(k1)~=idx(k2))
                        tn=tn+1;
                    end
                end
            end
        end
    end
end

p=tp/(tp+fp);
r=tp/(tp+fn);
f=2*p*r/(p+r);

end