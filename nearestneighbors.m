genres = readtable("Labels.csv");
features = readtable("extractedfeatures.csv");

rowidx = (genres.Var2 == "");
genres = genres(~rowidx, :);
[featuresidx] = genres.Var1(genres.Var1 <= 124911);
x = 1;
y = 1;
ind = [];

for i = 1:size(genres.Var1)
    if ismember(genres.Var1(i), features.Var1)
        ind = [ind find(features.Var1 == genres.Var1(i))];
    end
end

features = features(ind, :);
features = [features genres.Var2(ind)];

features = features(features.Var10 == "Rock" | features.Var10 == "Electronic", :);

labels = features.Var10;
acousticness = features.Var2;
danceability = features.Var3;
energy = features.Var4;
instrumentalness = features.Var5;
liveness = features.Var6;
speechiness	= features.Var7;
tempo = features.Var8;
valence = features.Var9;

X = [tempo speechiness];
Y = labels;

figure
ind0 = find(features.Var10 == "Rock");
ind1 = find(features.Var10 == "Electronic");

plot(X(ind0, 1), X(ind0, 2), 'ko', 'Color', "r"), hold on
plot(X(ind1, 1), X(ind1, 2), 'kx', 'Color', 'b');

hpartition = cvpartition(length(Y),'Holdout',0.3); % Nonstratified partition
idxTrain = training(hpartition);
Xtrain = X(idxTrain, :);
Ytrain = Y(idxTrain, :);
idxTest = test(hpartition);

model = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 3);
x1 = min(X(:,1)):0.1:max(X(:,1));
x2 = min(X(:,2)):0.001:max(X(:,2));

[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)];
pred = predict(model,XGrid);

figure
gscatter(XGrid(:,1),XGrid(:,2),pred)
hold on

ind0 = find(features.Var10 == "Rock");
ind1 = find(features.Var10 == "Electronic");

plot(X(ind0, 1), X(ind0, 2), 'ko')
plot(X(ind1, 1), X(ind1, 2), 'kx');


% [ind] = find(genres(:, 1) == tempo(:, 1));
% X = [danceability(ind, 2) tempo(ind, 2)];
% Y = genres(ind, 2);
% 
% model = fitcknn(X, Y);
% x1 = min(X(:,1)):0.01:max(X(:,1));
% x2 = min(X(:,2)):0.01:max(X(:,2));
% 
% [x1G,x2G] = meshgrid(x1,x2);
% XGrid = [x1G(:),x2G(:)];
% pred = predict(model,XGrid);
% 
% figure
% gscatter(XGrid(:,1),XGrid(:,2),pred,[1,0,0;0,0.5,1])
% hold on
% 
