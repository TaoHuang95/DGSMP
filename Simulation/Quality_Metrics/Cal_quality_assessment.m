clear all;clc;
gt_path = './Data/Testing_data/';

res_path = './Results/';

Test_file       =    {'scene01','scene02','scene03','scene04','scene05','scene06','scene07','scene08','scene09','scene10'};

res = zeros(6, length(Test_file));

for i=1:length(Test_file)
    S = load(strcat(gt_path, Test_file{i},'.mat'));
    S = S.img;

    Z = load(strcat(res_path, num2str(i),'.mat'));
    Z = double(Z.res);
    
    Z(Z>1.0) = 1.0;
    Z(Z<0.0) = 0.0;
    
    [psnr, rmse, ergas, sam, uiqi, ssim] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z)), 0, 1);
    
    res(1,i) = psnr;
    res(2,i) = rmse;
    res(3,i) = ergas;
    res(4,i) = sam;
    res(5,i) = uiqi;
    res(6,i) = ssim;
        
end
mean(res,2)
