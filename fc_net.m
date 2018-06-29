%
%�ɱ�����˹�������
classdef fc_net
    properties
        num_layers;   %ģ�Ͳ���
        parameters;   %ģ�����п�ѵ������
        use_dropout;  %�Ƿ�ʹ��dropout
        use_batchnorm;  %�Ƿ�ʹ��batchnorm
        grad_config;   %  adam�Ż����̵Ĳ���
    end
    methods
        %���캯������ʼ�����в���
        %layers   [ 1 x num_layers ] ����
        % p       ����,����dropout�ĸ��ʣ�0<=p<=1��С��0��Ϊ0������1��Ϊ1
        % use_batch_norm  ��������  ���Ƿ�ʹ��batch_norm  
        function obj=fc_net(layers,p,use_batchnorm)
            obj.parameters=containers.Map;
            obj.num_layers=length(layers);
            obj.grad_config=containers.Map;
            for i=1:length(layers)-1
                %xavier��ʼ������
                obj.parameters(strcat('w',num2str(i)))=normrnd(0,1.0,[layers(i),layers(i+1)])/sqrt(layers(i)/2);
                obj.parameters(strcat('b',num2str(i)))=zeros(1,layers(i+1));
                
                % adam ������ʼ��
                temp_grad_config={};
                temp_grad_config.beta1=0.9;
                temp_grad_config.beta2=0.999;
                temp_grad_config.m=zeros(layers(i),layers(i+1));
                temp_grad_config.v=zeros(layers(i),layers(i+1));
                temp_grad_config.t=1;
                obj.grad_config(strcat('w',num2str(i)))=temp_grad_config;
                temp_grad_config={};
                temp_grad_config.beta1=0.9;
                temp_grad_config.beta2=0.999;
                temp_grad_config.m=zeros(1,layers(i+1));
                temp_grad_config.v=zeros(1,layers(i+1));
                temp_grad_config.t=1;
                obj.grad_config(strcat('b',num2str(i)))=temp_grad_config;
                
            end
            % ����ģ���Ƿ�ʹ��dropout,batchnorm
            obj.use_dropout=(p>0);
            obj.use_batchnorm=use_batchnorm;
            if use_batchnorm
                
                for i=1:length(layers)-2
                    %��ʼ��batchnorm ����
                    obj.parameters(strcat('gamma',num2str(i)))=ones(1,layers(i+1));
                    obj.parameters(strcat('beta',num2str(i)))=zeros(1,layers(i+1));
                    %��ʼ�� adam����
                    temp_grad_config={};
                    temp_grad_config.beta1=0.9;
                    temp_grad_config.beta2=0.999;
                    temp_grad_config.m=zeros(1,layers(i+1));
                    temp_grad_config.v=zeros(1,layers(i+1));
                    temp_grad_config.t=1;
                    obj.grad_config(strcat('gamma',num2str(i)))=temp_grad_config;
                    temp_grad_config={};
                    temp_grad_config.beta1=0.9;
                    temp_grad_config.beta2=0.999;
                    temp_grad_config.m=zeros(1,layers(i+1));
                    temp_grad_config.v=zeros(1,layers(i+1));
                    temp_grad_config.t=1;
                    obj.grad_config(strcat('beta',num2str(i)))=temp_grad_config;
                end
            end
        end    
        
        %ǰ�򴫲�
        % x [a x b] ����
        % w [b x c] ����
        % b [1 x c] ����
        % result containers.Map
          % result('affine') [a x c] ����  ,���� x.dot(w)+b
          % result('cache') containers.Map �����򴫲��������
        function result=affine_forward(obj,x,w,b)
            result=containers.Map;
            cache=containers.Map;
            affine=x*w;
            size_affine=size(affine);
            affine=affine+repmat(b,size_affine(1),1);
            result('affine')=affine;            
            cache('x')=x;
            cache('w')=w;
            cache('b')=b;
            result('cache')=cache;
        end
        % ���򴫲�
        % upstream_grad �����ݶ�
        % cache ����affine_forward���ؽ���е�'cache'��
        % result contrainers.Map 
           % result('dx') x���ݶ���
           % result('dw') w���ݶ���
           % result('db') b���ݶ���
        function result=affine_backward(obj,upstream_grad,cache)
            result=containers.Map;
            temp_x=cache('x');
            temp_w=cache('w');
            temp_b=cache('b');
            dx=upstream_grad*temp_w';
            dw=temp_x'*upstream_grad;
            db=sum(upstream_grad,1);             
            result('dx')=dx;
            result('dw')=dw;
            result('db')=db;
        end
        %relu����� ǰ�򴫲�
        %input ����
        %result containers.Map
          % result('relu') ����������inputͬ��
          % result('cache') ���򴫲��������
        function result=relu_forward(obj,input)
            result=containers.Map;
            cache=containers.Map;
            relu=input;
            relu(relu<0)=0;
            result('relu')=relu;
            cache('input')=input;
            result('cache')=cache;
        end
        %relu����� ���򴫲�
        %upstream_grad �����ݶ�
        %cache relu_forward�����'cache'��
        %result  �ݶ���
        function result=relu_backward(obj,upstream_grad,cache)
            temp_input=cache('input');
            temp_input=(temp_input>0);
            temp_input=double(temp_input);
            result=upstream_grad.*temp_input;    
        end
        % batchnorm ǰ�򴫲�
        % input �������
        % gamma , beta   ���� batchnorm��ѵ����������
        % bn_param  struct  batchnorm����ѵ������
        % result contrainers.Map 
          % result('batchnorm') batchnorm ���
          % result('cache') ���򴫲��������
        function result=batchnorm_forward(obj,input,gamma,beta,bn_param)
            result=containers.Map;
            cache=containers.Map;
            mode=bn_param.mode;
            momentum=bn_param.momentum;
            running_mean=bn_param.running_mean;
            running_var=bn_param.running_var;
            size_input=size(input);
            batch_size=size_input(1);
            rep_gamma=repmat(gamma,batch_size,1);
            rep_beta=repmat(beta,batch_size,1);
            if strcmp(mode,'train')
                sample_mean=mean(input,1);
                sample_var=var(input,1,1);
                gaus_input=(input-repmat(sample_mean,batch_size,1))./sqrt(repmat(sample_var+1e-7,batch_size,1));
                result('batchnorm')=rep_gamma.*gaus_input+rep_beta;
                running_mean=momentum*running_mean+(1 - momentum)*sample_mean;
                running_var=momentum*running_var+(1 - momentum)*sample_var;
                cache('input')=input;
                cache('gaus_input')=gaus_input;
                cache('gamma')=gamma;
                cache('beta')=beta;
                cache('sample_mean')=sample_mean;
                cache('sample_var')=sample_var;
                result('cache')=cache;
            elseif strcmp(mode,'test')
                 rep_running_mean=repmat(running_mean,batch_size,1);
                 rep_running_var=repmat(running_var,batch_size,1);
                 result('batchnorm')=rep_gamma.*(input-rep_running_mean)./sqrt(rep_running_var+1e-7)+rep_beta;
                 result('cache')=cache;
            end
            bn_param.running_mean=running_mean;
            bn_param.running_var=running_var;
        end
        % batchnorm ���򴫲�
        % upstream_grad �����ݶ�
        % cache relu_forward�����'cache'��
        % result contrainers.Map
          % result('dx') �ݶ���
          %  result('dgamma') �ݶ���
          %  result('dbeta') �ݶ���
        function result=batchnorm_backward(obj,upstream_grad,cache)
            result=containers.Map;
            input=cache('input');
            gaus_input=cache('gaus_input');
            gamma=cache('gamma');
            beta=cache('beta');
            sample_mean=cache('sample_mean');
            sample_var=cache('sample_var');
            size_input=size(input);
            batch_size=size_input(1);
            rep_gamma=repmat(gamma,batch_size,1);
            rep_beta=repmat(beta,batch_size,1);
            rep_sample_mean=repmat(sample_mean,batch_size,1);
            rep_sample_var=repmat(sample_var,batch_size,1);
            temp_dx=upstream_grad.*rep_gamma;
            dsample_var=repmat(sum(temp_dx.*(input-rep_sample_mean),1),batch_size,1).*(-0.5*(rep_sample_var+1e-7).^(-1.5));
            dsample_mean=repmat(sum(-temp_dx./sqrt(rep_sample_var+1e-7),1),batch_size,1)+dsample_var.*repmat(sum(-2*(input-rep_sample_mean),1),batch_size,1)/batch_size;
            dx=temp_dx./sqrt(rep_sample_var+1e-7)+2*dsample_var.*(input-rep_sample_mean)/batch_size+dsample_mean/batch_size;
            dgamma=sum(upstream_grad.*gaus_input,1);
            dbeta=sum(upstream_grad,1);
            result('dx')=dx;
            result('dgamma')=dgamma;
            result('dbeta')=dbeta;
        end
        % dropout ǰ�򴫲�
        % input �������
        % dropout_param  struct  batchnorm����ѵ������
        % result contrainers.Map 
          % result('dropout') dropout ���
          % result('cache') ���򴫲��������
        function result=dropout_forward(obj,input,dropout_param)
            result=containers.Map;
            cache=containers.Map;
            p=dropout_param.p;
            mode=dropout_param.mode;
            if strcmp(mode,'train')
                mask=unifrnd(0,1,size(input));
                mask(mask<p)=0;
                mask(mask>=p)=1;
                mask=mask/p;
                result('dropout')=input.*mask;
                cache('mask')=mask;
                result('cache')=cache;
            elseif strcmp(mode,'test')
                result('dropout')=input;
            end
            
        end
        % dropout ���򴫲�
        % upstream_grad �����ݶ�
        % cache dropout_forward'cache'��
        % result �ݶ���
        function result=dropout_backward(obj,upstream_grad,cache)
                mask=cache('mask');
                result=upstream_grad.*mask;
        end
        % softmax ��ʧ����
        % scores ���� ǰһ�����
        % labels  ���� [1 x m]��ȷ���
        % reg ���� ���򻯲���
        % result containers.Map 
         % result('dscores') scores ���ݶ�
         % result('dw   +    ���� ')  ���򻯲������ݶ�
        function result=mysoftmax(obj,scores,labels,reg)
            result=containers.Map;
            size_scores=size(scores);
            batch_size=size_scores(1);
            num_category=size_scores(2);
            for i=1:batch_size
                correct_scores(i,1)=scores(i,labels(i));
            end
            correct_scores_repmat=repmat(correct_scores,1,num_category);
            temp_exp=exp(scores-correct_scores_repmat);
            temp_exp_sum=sum(temp_exp,2);
            temp_exp_sum_log=log(temp_exp_sum);
            loss=sum(temp_exp_sum_log)/batch_size;
            for i=1:obj.num_layers-1
                loss=loss+reg*sum(sum(obj.parameters(strcat('w',num2str(i)))));
            end
            result('loss')=loss;
            dscores=temp_exp./repmat(temp_exp_sum,1,num_category);
            for i=1:batch_size-1
                dscores(i,labels(i))=dscores(i,labels(i))-1;
            end
            dscores=dscores/batch_size;
            result('dscores')=dscores;
            for i=1:obj.num_layers-1
                result(strcat('dw',num2str(i)))=2*reg*obj.parameters(strcat('w',num2str(i)));
            end
        end
        % adam �ݶȷ��򴫲�
        function result=adam(obj,x,dx,learning_rate,grad_config)
            grad_config.t=(grad_config.t)+1;
            grad_config.m=grad_config.beta1.*grad_config.m+(1-grad_config.beta1).*dx;
            grad_config.v=grad_config.beta2.*grad_config.v+(1-grad_config.beta2).*dx.*dx;
            first_ui=grad_config.m./((1-grad_config.beta1).^grad_config.t);
            second_ui=grad_config.v./((1-grad_config.beta2).^grad_config.t);
            result=x-learning_rate*first_ui./sqrt(second_ui+1e-9);            
        end
        % ѵ��
        function loss=train(obj,dataset,labels,learning_rate,reg,dropout_param,bn_param)
            cache=containers.Map;
            affine_forward_result=containers.Map;
            batchnorm_forward_result=containers.Map;
            relu_forward_result=containers.Map;   
            drouput_forward_result=containers.Map; 
            relu_forward_result('relu')=dataset;
            for i=1:obj.num_layers-2
                affine_forward_result=obj.affine_forward(relu_forward_result('relu'),obj.parameters(strcat('w',num2str(i))),obj.parameters(strcat('b',num2str(i))));
                if obj.use_batchnorm
                    batchnorm_forward_result=obj.batchnorm_forward(affine_forward_result('affine'),obj.parameters(strcat('gamma',num2str(i))),obj.parameters(strcat('beta',num2str(i))),bn_param{i});
                else
                    batchnorm_forward_result('batchnorm')=affine_forward_result('affine');
                end
                relu_forward_result=obj.relu_forward(batchnorm_forward_result('batchnorm'));
                if obj.use_dropout
                    drouput_forward_result=obj.dropout_forward(relu_forward_result('relu'),dropout_param{i});          
                else
                    drouput_forward_result('dropout')=relu_forward_result('relu');
                end
                cache(strcat('affine',num2str(i)))=affine_forward_result;
                cache(strcat('batchnorm',num2str(i)))=batchnorm_forward_result;
                cache(strcat('relu',num2str(i)))=relu_forward_result;
                cache(strcat('dropout',num2str(i)))=drouput_forward_result;
            end
            scores=obj.affine_forward(drouput_forward_result('dropout'),obj.parameters(strcat('w',num2str(obj.num_layers-1))),obj.parameters(strcat('b',num2str(obj.num_layers-1))));
            softmax_result=obj.mysoftmax(scores('affine'),labels,reg);
            loss=softmax_result('loss');
            
            drouput_back_result=containers.Map;
            relu_back_result=containers.Map;
            batchnorm_back_result=containers.Map;
            affine_back_result=containers.Map;
            affine_back_result=obj.affine_backward(softmax_result('dscores'),scores('cache'));
%             obj.parameters(strcat('w',num2str(obj.num_layers-1)))=obj.parameters(strcat('w',num2str(obj.num_layers-1)))-learning_rate*affine_back_result('dw');
%             obj.parameters(strcat('w',num2str(obj.num_layers-1)))=obj.parameters(strcat('w',num2str(obj.num_layers-1)))-learning_rate*softmax_result(strcat('dw',num2str(obj.num_layers-1)));
%             obj.parameters(strcat('b',num2str(obj.num_layers-1)))=obj.parameters(strcat('b',num2str(obj.num_layers-1)))-learning_rate*affine_back_result('db');
            obj.parameters(strcat('w',num2str(obj.num_layers-1)))=obj.adam(obj.parameters(strcat('w',num2str(obj.num_layers-1))),affine_back_result('dw')+softmax_result(strcat('dw',num2str(obj.num_layers-1))),learning_rate,obj.grad_config(strcat('w',num2str(obj.num_layers-1))));
            obj.parameters(strcat('b',num2str(obj.num_layers-1)))=obj.adam(obj.parameters(strcat('b',num2str(obj.num_layers-1))),affine_back_result('db'),learning_rate,obj.grad_config(strcat('b',num2str(obj.num_layers-1))));
            for i=obj.num_layers-2:-1:1
                temp_dropout_cache=cache(strcat('dropout',num2str(i)));
                temp_relu_cache=cache(strcat('relu',num2str(i)));
                temp_batchnorm_cache=cache(strcat('batchnorm',num2str(i)));
                temp_affine_cache=cache(strcat('affine',num2str(i)));
                
                if obj.use_dropout 
                    dropout_back_result=obj.dropout_backward(affine_back_result('dx'),temp_dropout_cache('cache'));
                else 
                    dropout_back_result=affine_back_result('dx');
                end
                relu_back_result=obj.relu_backward(dropout_back_result,temp_relu_cache('cache'));
                if obj.use_batchnorm
                    batchnorm_back_result=obj.batchnorm_backward(relu_back_result,temp_batchnorm_cache('cache'));
                else
                    batchnorm_back_result('dx')=relu_back_result;
                end
                affine_back_result=obj.affine_backward(batchnorm_back_result('dx'),temp_affine_cache('cache'));
%                 obj.parameters(strcat('w',num2str(i)))=obj.parameters(strcat('w',num2str(i)))-learning_rate*affine_back_result('dw');
%                 obj.parameters(strcat('w',num2str(i)))=obj.parameters(strcat('w',num2str(i)))-learning_rate*softmax_result(strcat('dw',num2str(i)));
%                 obj.parameters(strcat('b',num2str(i)))=obj.parameters(strcat('b',num2str(i)))-learning_rate*affine_back_result('db');
                

                obj.parameters(strcat('w',num2str(i)))=obj.adam(obj.parameters(strcat('w',num2str(i))),affine_back_result('dw')+softmax_result(strcat('dw',num2str(i))),learning_rate,obj.grad_config(strcat('w',num2str(i))));
                obj.parameters(strcat('b',num2str(i)))=obj.adam(obj.parameters(strcat('b',num2str(i))),affine_back_result('db'),learning_rate,obj.grad_config(strcat('b',num2str(i))));




                if obj.use_batchnorm
%                     obj.parameters(strcat('gamma',num2str(i)))=obj.parameters(strcat('gamma',num2str(i)))-learning_rate*batchnorm_back_result('dgamma');
%                     obj.parameters(strcat('beta',num2str(i)))=obj.parameters(strcat('beta',num2str(i)))-learning_rate*batchnorm_back_result('dbeta');
                    obj.parameters(strcat('gamma',num2str(i)))=obj.adam(obj.parameters(strcat('gamma',num2str(i))),batchnorm_back_result('dgamma'),learning_rate,obj.grad_config(strcat('gamma',num2str(i))));
                    obj.parameters(strcat('beta',num2str(i)))=obj.adam(obj.parameters(strcat('beta',num2str(i))),batchnorm_back_result('dbeta'),learning_rate,obj.grad_config(strcat('beta',num2str(i))));

                end
            end
        end
        % Ԥ��
        function result=predict(obj,dataset,dropout_param,bn_param)
            affine_forward_result=containers.Map;
            batchnorm_forward_result=containers.Map;
            relu_forward_result=containers.Map;   
            drouput_forward_result=containers.Map; 
            relu_forward_result('relu')=dataset;
            for i=1:obj.num_layers-2
                affine_forward_result=obj.affine_forward(relu_forward_result('relu'),obj.parameters(strcat('w',num2str(i))),obj.parameters(strcat('b',num2str(i))));
                if obj.use_batchnorm
                    batchnorm_forward_result=obj.batchnorm_forward(affine_forward_result('affine'),obj.parameters(strcat('gamma',num2str(i))),obj.parameters(strcat('beta',num2str(i))),bn_param{i});
                else
                    batchnorm_forward_result('batchnorm')=affine_forward_result('affine');
                end
                relu_forward_result=obj.relu_forward(batchnorm_forward_result('batchnorm'));
                if obj.use_dropout
                    drouput_forward_result=obj.dropout_forward(relu_forward_result('relu'),dropout_param{i});          
                else
                    drouput_forward_result('dropout')=relu_forward_result('relu');
                end
            end
            scores=obj.affine_forward(drouput_forward_result('dropout'),obj.parameters(strcat('w',num2str(obj.num_layers-1))),obj.parameters(strcat('b',num2str(obj.num_layers-1))));
            [temp_notuse,result]=max(scores('affine'),[],2);
        end
    end
end
            
            