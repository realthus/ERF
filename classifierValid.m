function [accuracy] = classifierValid(trainX0,trainY0,trainX1,trainY1,trainX2,trainY2,testX0,testY0)

preAcc = [];
for i = 1:1
    model = fitctree(trainX0,trainY0);
    Predict = predict(model,testX0);
    classNum = size(unique(trainY0),1);
    finalaccuracy = size(find(Predict == testY0),1)/size(testY0,1);
    
    model = fitctree(trainX1,trainY1);
    Predict = predict(model,testX0);
    classNum = size(unique(trainY1),1);
    finalaccuracy = size(find(Predict == testY0),1)/size(testY0,1)+finalaccuracy;
    
    model = fitctree(trainX2,trainY2);
    Predict = predict(model,testX0);
    classNum = size(unique(trainY2),1);
    finalaccuracy = size(find(Predict == testY0),1)/size(testY0,1)+finalaccuracy;
    
    preAcc = [preAcc finalaccuracy];
end

accuracy = mean(preAcc);


end