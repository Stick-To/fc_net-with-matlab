% model ��װ �˹�������
classdef model
    properties
        network;  %fc_net ʵ��
        dropout_param;  %dropout ���� ����ӣ���Ӧд��fc_net��ѵ��������Ѻ����
        bn_param;       %dropout_param ���� ����ӣ���Ӧд��fc_net��ѵ��������Ѻ����
    end
    methods
        % model ���캯��
        % layers [1 x num_layers] ����
        % p       ����,����dropout�ĸ��ʣ�0<=p<=1��С��0��Ϊ0������1��Ϊ1
        % use_batch_norm  ��������  ���Ƿ�ʹ��batch_norm
        function obj=model(layers,p,use_batchnorm)
            obj.network=fc_net(layers,p,use_batchnorm);
            obj.dropout_param={};
            obj.bn_param={};
            for i=1:length(layers)-2
                
                temp_dropout_param.mode='train';
                temp_dropout_param.p=p;        
                obj.dropout_param{i}=temp_dropout_param;
                temp_bn_param.mode='train';
                temp_bn_param.momentum=0.9;
                temp_bn_param.running_mean=zeros(1,layers(i+1));
                temp_bn_param.running_var=zeros(1,layers(i+1));
                obj.bn_param{i}=temp_bn_param;
            end
        end
        %Ԥ��
        function result=predict(obj,dataset,dropout_pa,bn_pa)
            result=obj.network.predict(dataset,dropout_pa,bn_pa);
        end
        % ���׼ȷ��
        function result=get_accuracy(obj,pred,labels)
            temp_acc=(pred==labels);
            size_pred=size(pred);
            result=sum(double(temp_acc))/size_pred(1);
        end
        % ѵ��
        % trainset ѵ����
        % trainlabels ѵ�������
        % learning_rate ���Լ�
        % reg  ���򻯲���
        % iterations ��������
        % batch_size 
        % valset ��֤�� 
        % vallabels ��֤�����
        % print_num  ÿ�������ٲ���ӡһ����֤���
        function train(obj,trainset,trainlabels,learning_rate,reg,iterations,batch_size,valset,vallabels,print_num)
            lr=learning_rate;
            size_trainset=size(trainset);
            num_train=size_trainset(1);
            iterator_per_epoch=floor(num_train/batch_size);          
            shuffle=randperm(num_train);
            shuffle_trainset=trainset(shuffle,:);
            shuffle_trainlabels=trainlabels(shuffle);
            j=1;
            for i=1:iterations
                if mod(i,iterator_per_epoch)==0
                    j=1;
                    shuffle=randperm(num_train);
                    shuffle_trainset=trainset(shuffle,:);
                    shuffle_trainlabels=trainlabels(shuffle);
                end
                sub_shuffle_trainset=shuffle_trainset(batch_size*(j-1)+1:batch_size*j,:);
                sub_shuffle_trainlabels=shuffle_trainlabels(batch_size*(j-1)+1:batch_size*j);
                loss=obj.network.train(sub_shuffle_trainset,sub_shuffle_trainlabels,lr,reg,obj.dropout_param,obj.bn_param);
                j=j+1;
                
                
                
%                 if i==50000
%                     lr=5e-4;
%                 end
%                 if i==10000
%                     lr=lr*0.5;
%                 end
%                 if i==20000
%                     lr=lr*0.5;
%                 end
%                 
%                 
            
          
                if mod(i,print_num) ==0
                    for m=1:obj.network.num_layers-2
                        obj.dropout_param{m}.mode='test';
                        obj.bn_param{m}.mode='test';
                    end
                    batch_pred=obj.predict(sub_shuffle_trainset,obj.dropout_param,obj.bn_param);
                    batch_accuracy=obj.get_accuracy(batch_pred,sub_shuffle_trainlabels');
                    val_pred=obj.predict(valset,obj.dropout_param,obj.bn_param);
                    val_accuracy=obj.get_accuracy(val_pred,vallabels');
                    
                    fprintf('iterators:%d train batch loss:%f\n',i,loss);
                    fprintf('learning rate=%f\n',lr);
                    fprintf('train batch accuracy:%f\n',batch_accuracy);
                    fprintf('val set accuracy:%f\n',val_accuracy);
                    for m=1:obj.network.num_layers-2
                        obj.dropout_param{m}.mode='train';
                        obj.bn_param{m}.mode='train';
                    end
                end
               
            end
        end
    end
end