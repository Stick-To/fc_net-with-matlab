% 测试函数
% model 模型
% testset 测试集
% testlabels 测试集类标
function [test_pred,test_accuracy]=test(model,testset,testlabels)
    for m=1:model.network.num_layers-2
                        model.dropout_param{m}.mode='test';
                        model.bn_param{m}.mode='test';
    end
    
    test_pred=model.predict(testset,model.dropout_param,model.bn_param);
    test_accuracy=model.get_accuracy(test_pred,testlabels');
    
    
    for m=1:model.network.num_layers-2
                        obj.dropout_param{m}.mode='train';
                        obj.bn_param{m}.mode='train';
    end
end