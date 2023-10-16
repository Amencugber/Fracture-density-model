%% ����Tentӳ�����Ⱥ��ʼ���Ӻ���
function Positions=tentInitialization(popsize,dim,ub,lb)
%input��popsize   ��Ⱥ����
%             dim           ����ά��
%             ub             ��������
%             lb               ��������
%return��Positions    ���ɵĳ�ʼ��Ⱥλ��

%��ʼ��λ��0����
Positions=zeros(popsize,dim);

%��ÿ�����壬����ӳ�����λ��
for i = 1:popsize
    value =  Tent(dim);  %����ӳ������
    Positions(i,:)=value.*(ub-lb)+lb;
    %λ��Խ������
    Positions(i,:)=min(Positions(i,:),ub);   %�Ͻ����
    Positions(i,:)=max(Positions(i,:),lb);    %�½����
end

end



%����ӳ���Ӻ���
function sequence=Tent(n)
%input��n    �������г���
%return��value    ���ɵĻ�������

%��ʼ������
sequence=zeros(1,n);    
sequence(1)=rand; %�������
beta = 0.7;%����beta��ȡֵ��Χ(0,1)
for i=1:n-1
    if sequence(i)<beta
        sequence(i+1)=sequence(i)/beta;
    end
    if sequence(i)>=beta
        sequence(i+1)=(1-sequence(i))/(1-beta);
    end
end

end




