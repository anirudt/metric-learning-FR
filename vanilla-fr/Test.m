%% Loading the training dataset 
load('model.mat','mean_face','evectors','filenames','features','num_images','image_dims','images');
load('kernel9.mat');
%% loading the testing data (images) 
path='C:\Users\shubho\Desktop\New folder (2)\PCA EXPERIMENT EFFICIENCY\test\';
for i=1:117
        filename=[path,'W (',int2str(i),').jpg'];
        input_image_1=imread(filename);
        input_image = imfilter(input_image_1,H);
%% calculate the similarity of the input to each training image
        feature_vec = evectors' * (double(input_image(:)) - mean_face);
        similarity_score = arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)), 1:num_images);
 
%% find the image with the highest similarity
        [match_score, match_ix] = max(similarity_score);
 
%% display the result
% hold on
%         subplot(4,4,i)
%         imshow([input_image reshape(images(:,match_ix), image_dims)]);
%         title(sprintf('matches %s, score %f', filenames(match_ix).name, match_score));
% temp = filenames(match_ix).name;
        chrNumeric = uint16(filenames(match_ix).name);
        
        if length(chrNumeric) == 9
            chrAlpha = str2num(char([chrNumeric(1,4)]));
        elseif  length(chrNumeric) == 10   
            chrAlpha = str2num(char([chrNumeric(1,4)  chrNumeric(1,5)]));
        else
            chrAlpha = str2num(char([chrNumeric(1,4)  chrNumeric(1,5)  chrNumeric(1,6)]));
        end

        testclass = ceil(i/3);
        trainclass = ceil(chrAlpha/7);

            if testclass == trainclass
            	efficiency(i,1) = 1;
            else
                efficiency(i,1) = 0;  
          end

end
 
performance = (sum(efficiency)*100)/117
