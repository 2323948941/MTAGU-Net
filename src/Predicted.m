%% prediction
YPrediction=cell(test_num,1);
[n1,n2,n3,n4,n5]=size(XTest);
XTest1=zeros(n1,n2,n3,1,test_num);
YTest2=cell(test_num,1);
for i=1:test_num
    YPrediction{i} = predict(MTAGU_Net_struct, XTest(:,:,:,:,i));  %1   层网络预测
    YPrediction{i}=double(YPrediction{i});
    YTest2{i}=YTest(:,:,:,:,i);
end
% Anti-normalization
for i=1:test_num

    YTest2{i}=fgyh(YTest2{i},max_output);
    YPrediction{i}=fgyh(YPrediction{i},max_output);
end

Rmse1=zeros(test_num,1);
for i =1:test_num
    Rmse1(i)= RMSE(YTest2{i},YPrediction{i});
end
[sort_RMse1,inx2]=sort(Rmse1,'ascend');


%% 3D visualization
c=ones(1,22);
b=ones(1,22);
a=ones(1,22);
cc=[0 cumsum(c)];
bb=[0 cumsum(b)]-sum(b,2)/2;
aa=[0 cumsum(a)]-sum(a,2)/2;
ccz=cc(2:end)-c/2;
ccz= ccz(end:-1:1);
aax=aa(2:end)-a/2;
bby=bb(2:end)-b/2;
f=log10(logspace(-1,3,16));
dyy=bby;
dxx=aax;
dy=dyy;
dx=dxx;
[xx,yy,ff]=meshgrid(dyy,dxx,f);

for j=2:5
    figure(j)
    k=1;
    for i=1:2:7
        subplot(4,2,i)
        h1=slice(xx,yy,ff,(YTest2{j}(:,:,:,k)),dx,dy,df);
        hold on
        alpha (0.8)
        shading interp
        hcb=colorbar;

        set(get(hcb,'Title'),'string','log_1_0(\Omega\cdot m)');
        set(hcb,'FontSize',12);
        colormap jet
        xlabel('y')
        ylabel('x')
        zlabel('f')
        subplot(4,2,i+1)
        h1=slice(xx,yy,ff,(YPrediction{j}(:,:,:,k)),dx,dy,df);
        hold on
        alpha (0.8)
        shading interp
        hcb=colorbar;
        set(get(hcb,'Title'),'string','log_1_0(\Omega\cdot m)');
        set(hcb,'FontSize',12);
        colormap jet
        xlabel('X(km)')
        ylabel('Y(km)')
        zlabel('f(Hz)')
        k=k+1;

    end
end
%% frequency slice diagram
close all
for k=2:10
    i=inx2(k);
    kk=1;
    m1=1;
    figure
    for j =9:16
        subplot(8,3,kk)
        h1=contourf(dx,dy,(YTest2{i}(:,:,j,m1)));    hold on
        contour(dx,dy,(YTest2{i}(:,:,j,m1)),'w');

        shading interp
        hcb=colorbar;
        set(get(hcb,'Title'),'string','log_1_0(\Omega\cdot m)');
        set(hcb,'FontSize',12);
        colormap jet
        xlabel('y')
        ylabel('x')
        zlabel('z')

        subplot(8,3,kk+1)
        h1=contourf(dx,dy,(YPrediction{i}(:,:,j,m1)));    hold on
        contour(dx,dy,(YPrediction{i}(:,:,j,m1)),'w');
        shading interp
        hcb=colorbar;
        set(get(hcb,'Title'),'string','log_1_0(\Omega\cdot m)');
        set(hcb,'FontSize',12);
        colormap jet
        xlabel('X(km)')
        ylabel('Y(km)')

        subplot(8,3,kk+2)
        wucha=YTest2{i}(:,:,j,m1)-YPrediction{i}(:,:,j,m1);
        h1=contourf(dx,dy,wucha);    hold on
        contour(dx,dy,wucha,'w');
        shading interp
        hcb=colorbar;
        set(get(hcb,'Title'),'string','log_1_0(\Omega\cdot m)');
        set(hcb,'FontSize',12);
        colormap jet
        xlabel('X(km)')
        ylabel('Y(km)')
        kk=kk+3;
    end
end


