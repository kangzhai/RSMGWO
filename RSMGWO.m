function [Alpha_score,Cbest,Alpha_pos]=RSMGWO(f,lb,ub,d,M,N)
%Improved GWO for large-scale function optimization and MLP optimization in cancer identification(RSMGWO)
%Programmer: Xinming Zhang
%Date:2018-6-26
% initialize alpha, beta, and delta_pos    //初始化α、β和δ狼
Alpha_pos=zeros(1,d);
Alpha_score=inf; %change this to -inf for maximization problems
Beta_pos=zeros(1,d);
Beta_score=inf; %change this to -inf for maximization problems
Delta_pos=zeros(1,d);
Delta_score=inf; %change this to -inf for maximization problems
%Initialize the positions of search agents
aa=repmat(lb,N,1);bb=repmat(ub,N,1);
Positions=aa+(bb-aa).*rand(N,d);%fitval=zeros(1,size(Positions,1));
X=Positions;
X(:,d+1)=feval(f, X(:,1:d));
X = PopSort(X);
Positions=X(:,1:d);s1=1;s2=N+1;s3=N+1;
fit=X(:,d+1);
for i=1:N
    if fit(i)<Alpha_score
        Alpha_score=fit(i); % Update alpha
        Alpha_pos=Positions(i,:);s1=i;
    end
    if fit(i)>Alpha_score && fit(i)<Beta_score
        Beta_score=fit(i); % Update beta
        Beta_pos=Positions(i,:);s2=i;
    end
    if fit(i)>Alpha_score && fit(i)>Beta_score && fit(i)<Delta_score
        Delta_score=fit(i); % Update delta
        Delta_pos=Positions(i,:);s3=i;
    end
end
Cbest=zeros(1,M);
Cbest(1)=Alpha_score;
a1=2; 
l=0;% Loop counter
% Main loop
while l<M
    a=2-l*((2)/M); % a decreases linearly fron 2 to 0
    % EPD
    [~,index]=sort(fit);
    Positions=Positions(index,:);%
    Num=round(N/4);
    for i=1:Num%采用EPD
        pa=rand;
        if pa<=0.25
            temp=Alpha_pos;
        elseif pa<=0.5
            temp=Beta_pos;
        elseif pa<=0.75
            temp=Delta_pos;
        else
            temp=[];
        end
        if isempty(temp)
            Positions(N-Num+i,1:d)=lb+(ub-lb).*rand(1,d);
        else
            if rand<0.5
                Positions(N-Num+i,1:d)=temp+(ub-temp).*rand(1,d)*ba;
            else
                Positions(N-Num+i,1:d)=temp-(temp-lb).*rand(1,d)*ba;
            end
        end
    end
    Num=ceil(N*rand);while Num==s1||Num==s2||Num==s3,Num=ceil(N*rand);end%随机选择一个灰狼
    for i=1:N
        if i==Num
            Positions(i,:)=lb+(ub-Alpha_pos);%反向学习
        elseif i==s1%对于alpha狼
            for j=1:d
                cnum=ceil(d*rand);
                if cnum~=j
                    Positions(i,j)=Alpha_pos(cnum);
                else
                    rnum=ceil(N*rand);while rnum==i,rnum=ceil(N*rand);end;
                    rnum1=ceil(N*rand);while rnum1==i || rnum1==rnum,rnum1=ceil(N*rand);end;
                    if rand<=0.5
                        Positions(i,j)=Positions(i,j)+3.5*a*rand*(Positions(rnum,j)-Positions(rnum1,j));
                    else
                        Positions(i,j)=Beta_pos(j)-a*(2*rand-1)*abs(a1*rand*Beta_pos(j)-Positions(i,j));
                    end
                end
            end
        elseif i==s2%对于beta狼
            cc=rand(1,d)>=0.67;snum=sum(cc);
            Positions(i,cc)=Alpha_pos(cc)-a*(2*rand(1,snum)-1).*abs(a1*rand(1,snum).*Alpha_pos(cc)-Positions(i,cc));
            snum=d-snum;
            cnum=ceil(rand(1,snum)*d);
            Positions(i,~cc)=(Alpha_pos(cnum)+Beta_pos(cnum))/2;
        elseif i==s3%对于delta狼
            cc=rand(1,d)>=0.33;snum=sum(cc);
            X1=Alpha_pos(cc)-a*(2*rand(1,snum)-1).*abs(a1*rand(1,snum).*Alpha_pos(cc)-Positions(i,cc));
            X2=Beta_pos(cc)-a*(2*rand(1,snum)-1).*abs(a1*rand(1,snum).*Beta_pos(cc)-Positions(i,cc));
            Positions(i,cc)=(X1+X2)/2;
            snum=d-snum;
            cnum=ceil(d*rand(1,snum));
            Positions(i,~cc)=(Alpha_pos(cnum)+Beta_pos(cnum)+Delta_pos(cnum))/3;
        Else%对于其它狼
            X1=Alpha_pos-a*(2*rand(1,d)-1).*abs(a1*rand(1,d).*Alpha_pos-Positions(i,:));
            X2=Beta_pos-a*(2*rand(1,d)-1).*abs(a1*rand(1,d).*Beta_pos-Positions(i,:));
            X3=Delta_pos-a*(2*rand(1,d)-1).*abs(a1*rand(1,d).*Delta_pos-Positions(i,:));
            Positions(i,:)=(X1+X2+X3)/3;
        end
    end
    Positions=ControlBound(Positions,aa,bb);%限定位置
    fit=feval(f,Positions);%求适应度值
    for i=1:N
        % Calculate objective function for each search agent
        fitness=fit(i);        
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);s1=i;
        end
        if fitness>Alpha_score && fitness<Beta_score
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);s2=i;
        end
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);s3=i;
        end
    end
    l=l+1;
    Cbest(l)=Alpha_score;
end

