function [Accuracy,Std] = CalculayteAccCrossTest(sample_Division1,col,model10,Ri10,C,L,flag)

Test = 10;
classification = 2;
trainData = cell(1,3);
testData = cell(1,3);
ConfidenceAccuracy1 = [];
for T = 1:Test
    Data = sample_Division1{T,col};
    trainData{1,1} = full(Data{1,1});
    trainData{1,2} = full(Data{1,2});
    trainData{1,3} = full(Data{1,3});
    
    testData{1,1} = full(Data{3,1});
    testData{1,2} = full(Data{3,2});
    testData{1,3} = full(Data{3,3});
    
    models = model10{T,1};
    Ri = Ri10{T,1};
    preResult = [];
    subspaceAcc = [];
    classNum = size(unique(trainData{1,1}(:,end)),1);
    preConfidence = zeros(size(testData{1,1},1),classNum);
    w = [1 1 1];
    for layer = 1:L
        lC = C;
        wi = w(:,layer);
        for c = 1:lC
            if(flag == 1)
                testZ = testData{1,1}(:,1:end-1)*Ri{layer,c};
            else
                testZ = testData{1,layer}(:,1:end-1)*Ri{layer,c};
            end
            testY = testData{1,1}(:,end);
            if(classification == 1)
                Predict = svmpredict(testY,testZ,models{layer,c});
            elseif(classification == 2)
                [Predict,Confidence] = predict(models{layer,c},testZ);
            end
            preAcc = calculateAcc(Predict,testY);
            subspaceAcc = [subspaceAcc preAcc];
            preResult = [preResult Predict];
            preConfidence = preConfidence + Confidence*wi;
        end
    end
    preConfidenceResultFinal = [];
    [max_a,preConfidenceResultFinal] = max(preConfidence/(C*L),[],2);
    uni = unique(trainData{1,1}(:,end));
    preConfilabel = uni(preConfidenceResultFinal(:,1),1);
    ConfidenceFinalAccuracy1 = size(find(preConfilabel == testData{1,1}(:,end)),1)/size(testData{1,1}(:,end),1);
    ConfidenceAccuracy1 = [ConfidenceAccuracy1 ConfidenceFinalAccuracy1];
end
Accuracy = mean(ConfidenceAccuracy1);
Std = std(ConfidenceAccuracy1);
end