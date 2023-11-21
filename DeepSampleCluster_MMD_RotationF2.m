function [models ,Ri] = DeepSampleCluster_MMD_RotationF2(trainData,c,L,classification)

K =floor((size(trainData,2)-1)/3);
if(K==0)
    K=1;
end
models = cell(L,c);
Ri = cell(L,c);

for layer = 1:L
    trainSub = trainData{1,layer};
    trainSubX = trainSub(:,1:end-1);
    trainSubY = trainSub(:,end);
    lc = c;
    for ci = 1:lc
        [trainX_new ,trainY_new , Ri_order] = RotationForest2(trainSubX,trainSubY,K);
        if(classification == 1)
            model = svmtrain(trainY_new,trainX_new,'-s 0 -t 0 ');
        elseif(classification == 2)
            model = fitctree(trainX_new,trainY_new);
        end
        Ri{layer,ci} = Ri_order;
        models{layer,ci} = model;
    end

end


end
