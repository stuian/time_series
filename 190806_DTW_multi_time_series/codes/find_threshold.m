function threshold = find_threshold()

threshold=1;
iSuccess=load('iSuccess.txt');
dSuccess=load('dSuccess.txt');
if (isempty(iSuccess) && isempty(dSuccess))
    threshold= 1;
elseif (isempty(iSuccess) && ~isempty(dSuccess))
    threshold=max(dSuccess)+0.1;
elseif (~isempty(iSuccess) && isempty(dSuccess))
    threshold=min(iSuccess)-0.1;
else
    %threshold=dSuccess(1,10);
    common=size(iSuccess,1)+size(dSuccess,1);
    iFlag=0;
    dFlag=0;
    if (size(iSuccess,1)>0)
        iFlag=1;
    end
    if (size(dSuccess,1)>0)
        dFlag=1;
    end
    
    if (dFlag)
        for j=1:size(dSuccess,1)
            I=0;
            D=0;
            for i=1:size(dSuccess,1)
                if(dSuccess(i,10)>= dSuccess(j,10))
                    D=D+1;
                end
            end
            if (iFlag)
                for i=1:size(iSuccess,1)
                    if(iSuccess(i,10)< dSuccess(j,10))
                        I=I+1;
                    end
                end
            end
            if ((I+D)<common)
                common=I+D;
                threshold=dSuccess(j,10);
            end
        end
    end
    
    if (iFlag)
        for j=1:size(iSuccess,1)
            I=0;
            D=0;
            if (dFlag)
                for i=1:size(dSuccess,1)
                    if(dSuccess(i,10)>= iSuccess(j,10))
                        D=D+1;
                    end
                end
            end
            for i=1:size(iSuccess,1)
                if(iSuccess(i,10)< iSuccess(j,10))
                    I=I+1;
                end
            end
            if ((I+D)<common)
                common=I+D;
                threshold=iSuccess(j,10);
            end
        end
    end
    
end
end