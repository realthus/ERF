function [ acc ] = calculateAcc(testY,label)
if(size(testY,1) ~= size(label,1))
    fprintf("����������ά�Ȳ�ƥ��");
end

acc = size(find(testY==label),1)/size(label,1);
end