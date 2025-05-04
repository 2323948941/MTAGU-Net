lgraph = layerGraph();
tempLayers = [
    image3dInputLayer([32 32 32 1],"Name","ImageInputLayer","Normalization","none")
    convolution3dLayer([3 3 3],64,"Name","Encoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-1-BN-1")
    reluLayer("Name","Encoder-Stage-1-ReLU-1")
    convolution3dLayer([3 3 3],128,"Name","Encoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-1-BN-2")
    reluLayer("Name","Encoder-Stage-1-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-1-MaxPool","Stride",[2 2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([2 2 2],64,"Name","Encoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-2-BN-1")
    reluLayer("Name","Encoder-Stage-2-ReLU-1")
    convolution3dLayer([2 2 2],128,"Name","Encoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-2-BN-2")
    reluLayer("Name","Encoder-Stage-2-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    convolution3dLayer([3 3 3],256,"Name","Encoder-Stage-2-Conv-2_1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-2-BN-2_1")
    reluLayer("Name","Encoder-Stage-2-ReLU-2_1")
    maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-2-MaxPool","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],128,"Name","Encoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-3-BN-1")
    reluLayer("Name","Encoder-Stage-3-ReLU-1")
    convolution3dLayer([3 3 3],256,"Name","Encoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-3-BN-2")
    reluLayer("Name","Encoder-Stage-3-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    convolution3dLayer([3 3 3],512,"Name","Bridge-Conv-1_1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Bridge-BN-1_1")
    reluLayer("Name","Bridge-ReLU-1_1")
    maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-3-MaxPool","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],256,"Name","Bridge-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Bridge-BN-1")
    reluLayer("Name","Bridge-ReLU-1")
    convolution3dLayer([3 3 3],512,"Name","Bridge-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Bridge-BN-2")
    reluLayer("Name","Bridge-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    convolution3dLayer([3 3 3],512,"Name","Bridge-Conv-2_1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Bridge-BN-2_1")
    reluLayer("Name","Bridge-ReLU-2_1")
    transposedConv3dLayer([2 2 2],512,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"Cropping","same","Stride",[2 2 2],"WeightsInitializer","he")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop3dLayer("Name","Crop3d-1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","Decoder-Stage-1-Concatenation")
    convolution3dLayer([3 3 3],512,"Name","Decoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-1-BN-1")
    reluLayer("Name","Decoder-Stage-1-ReLU-1")
    convolution3dLayer([3 3 3],256,"Name","Decoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-1-BN-2")
    reluLayer("Name","Decoder-Stage-1-ReLU-2")
    transposedConv3dLayer([2 2 2],256,"Name","Decoder-Stage-2-UpConv","BiasLearnRateFactor",2,"Stride",[2 2 2],"WeightsInitializer","he")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop3dLayer("Name","Crop3d-2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","Decoder-Stage-2-Concatenation")
    convolution3dLayer([3 3 3],256,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-2-BN-1")
    reluLayer("Name","Decoder-Stage-2-ReLU-1")
    convolution3dLayer([3 3 3],128,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-2-BN-2")
    reluLayer("Name","Decoder-Stage-2-ReLU-2")
    transposedConv3dLayer([2 2 2],128,"Name","Decoder-Stage-3-UpConv","BiasLearnRateFactor",2,"Stride",[2 2 2],"WeightsInitializer","he")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop3dLayer("Name","Crop3d-3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","Decoder-Stage-3-Concatenation")
    convolution3dLayer([3 3 3],128,"Name","Decoder-Stage-3-Conv-1","Padding","same","Stride",[1 1 2],"WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-3-BN-1")
    reluLayer("Name","Decoder-Stage-3-ReLU-1")
    convolution3dLayer([3 3 3],64,"Name","Decoder-Stage-3-Conv-2_1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-3-BN-2")
    reluLayer("Name","Decoder-Stage-3-ReLU-2")
    convolution3dLayer([3 3 3],4,"Name","Decoder-Stage-3-Conv-2_2","Padding","same","WeightsInitializer","he")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);


%Connect all branches of the network to create a network diagram.
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Crop3d-3/in");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-MaxPool","Encoder-Stage-2-Conv-1");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-MaxPool","addition_2/in1");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","addition_2/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Crop3d-2/in");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-MaxPool","Encoder-Stage-3-Conv-1");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-MaxPool","addition/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","addition/in1");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Crop3d-1/in");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-MaxPool","Bridge-Conv-1");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-MaxPool","addition_1/in1");
lgraph = connectLayers(lgraph,"Bridge-ReLU-2","addition_1/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpConv","Crop3d-1/ref");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpConv","Decoder-Stage-1-Concatenation/in1");
lgraph = connectLayers(lgraph,"Crop3d-1","Decoder-Stage-1-Concatenation/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpConv","Crop3d-2/ref");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpConv","Decoder-Stage-2-Concatenation/in1");
lgraph = connectLayers(lgraph,"Crop3d-2","Decoder-Stage-2-Concatenation/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpConv","Crop3d-3/ref");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpConv","Decoder-Stage-3-Concatenation/in1");
lgraph = connectLayers(lgraph,"Crop3d-3","Decoder-Stage-3-Concatenation/in2");