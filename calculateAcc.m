function [ acc ] = calculateAcc(testY,label)
if(size(testY,1) ~= size(label,1))
    fprintf("两个列向量维度不匹配");
end

acc = size(find(testY==label),1)/size(label,1);
end