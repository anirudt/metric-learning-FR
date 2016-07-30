%% File capturing form the training folder and normalising the image - dimensionally and photometrically 
clear all;close all;clc;
input_dir = 'C:\Users\shubho\Desktop\New folder (2)\PCA EXPERIMENT EFFICIENCY\train\';
image_dims = [100, 100];
filenames = dir(fullfile(input_dir, '*.jpg'));
num_images = numel(filenames);
images = [];
for n = 1:num_images
    filename = fullfile(input_dir, filenames(n).name);
    img = imread(filename);
%     if n == 1
%         images = zeros(prod(image_dims), num_images);
%     end
    images(:, n) = img(:);
end
%% Find the mean image and the mean-shifted input images
mean_face = mean(images, 2);
shifted_images = images - repmat(mean_face, 1, num_images);
 
%% Calculate the ordered eigenvectors and eigenvalues
[evectors,evalues,score] = pca(images');
 
%% Only retain the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
num_eigenfaces = 70;
evectors = evectors(:, 1:num_eigenfaces);
 
%% Project the images into the subspace to generate the feature vectors
features = evectors' * shifted_images;
save('model.mat','mean_face','evectors','features','num_images','filenames','image_dims','images');

% %% Normalization of eigenvectors
% for i=1:num_eigenfaces
%    kk=evectors(:,i);
%    temp=sqrt(sum(kk.^2));
% 	evectors(:,i)=evectors(:,i)./temp;
% end
% 
% %% Show eigenfaces;
% for i=1:num_eigenfaces
%     img=reshape(evectors(:,i),100,100);
%     img=img';
%     img=histeq(img,255);
%     subplot(ceil(sqrt(20)),ceil(sqrt(20)),i)
%     imshow(img')
%     drawnow;
%     if i==3
%         title('Eigenfaces','fontsize',18)
%     end
% end
