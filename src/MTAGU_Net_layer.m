lgraph = layerGraph();
tempLayers = [
    image3dInputLayer([22 22 22 1],"Name","encoderImageInputLayer")
    resize3dLayer("Name","resize3d-output-size_1","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[32 32 32])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],32,"Name","Encoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-1-BN-1")
    reluLayer("Name","Encoder-Stage-1-ReLU-1")
    convolution3dLayer([3 3 3],64,"Name","Encoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-1-BN-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],64,"Name","Encoder-Stage-1-Conv-2_1","Padding","same","WeightsInitializer","he");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_9")
    reluLayer("Name","Encoder-Stage-1-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],64,"Name","conv3d_1","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-1-MaxPool","Stride",[2 2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],128,"Name","Encoder-Stage-2-Conv-1_1","Padding","same","WeightsInitializer","he");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],128,"Name","Encoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-2-BN-1")
    reluLayer("Name","Encoder-Stage-2-ReLU-1")
    convolution3dLayer([3 3 3],128,"Name","Encoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-2-BN-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3")
    reluLayer("Name","Encoder-Stage-2-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-2-MaxPool","Stride",[2 2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],256,"Name","Encoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-3-BN-1")
    reluLayer("Name","Encoder-Stage-3-ReLU-1")
    convolution3dLayer([3 3 3],256,"Name","Encoder-Stage-3-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Encoder-Stage-3-BN-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],256,"Name","conv3d_4","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],256,"Name","Encoder-Stage-3-Conv-2_1","Padding","same","WeightsInitializer","he");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    reluLayer("Name","Encoder-Stage-3-ReLU-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],512,"Name","conv3d_7","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    dropoutLayer(0.1,"Name","Encoder-Stage-3-DropOut")
    maxPooling3dLayer([2 2 2],"Name","Encoder-Stage-3-MaxPool","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],512,"Name","LatentNetwork-Bridge-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","LatentNetworkBridge-BN-1")
    reluLayer("Name","LatentNetwork-Bridge-ReLU-1")
    convolution3dLayer([3 3 3],512,"Name","LatentNetwork-Bridge-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","LatentNetworkBridge-BN-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],512,"Name","LatentNetwork-Bridge-Conv-2_1","Padding","same","WeightsInitializer","he");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5")
    reluLayer("Name","LatentNetwork-Bridge-ReLU-2")
    dropoutLayer(0.2,"Name","LatentNetwork-Bridge-DropOut")
    transposedConv3dLayer([2 2 2],512,"Name","Decoder-Stage-1-UpConv","BiasLearnRateFactor",2,"Stride",[2 2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-1-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],512,"Name","conv3d_6","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    reluLayer("Name","relu_2")
    convolution3dLayer([1 1 1],512,"Name","conv3d_8","Padding","same")
    sigmoidLayer("Name","sigmoid_2")
    transposedConv3dLayer([1 1 1],512,"Name","transposed-conv3d_2","Cropping","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop3dLayer("Name","encoderDecoderSkipConnectionCrop3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,2,"Name","encoderDecoderSkipConnectionFeatureMerge3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],256,"Name","Decoder-Stage-1-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-1-BN-1")
    reluLayer("Name","Decoder-Stage-1-ReLU-1")
    dropoutLayer(0.3,"Name","LatentNetwork-Bridge-DropOut_1")
    convolution3dLayer([3 3 3],256,"Name","Decoder-Stage-1-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-1-BN-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],256,"Name","Decoder-Stage-1-Conv-2_1","Padding","same","WeightsInitializer","he");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_6")
    reluLayer("Name","Decoder-Stage-1-ReLU-2")
    transposedConv3dLayer([2 2 2],256,"Name","Decoder-Stage-2-UpConv","BiasLearnRateFactor",2,"Stride",[2 2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-2-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],256,"Name","conv3d_3","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    reluLayer("Name","relu_1")
    convolution3dLayer([1 1 1],256,"Name","conv3d_5","Padding","same")
    sigmoidLayer("Name","sigmoid_1")
    transposedConv3dLayer([1 1 1],256,"Name","transposed-conv3d_1","Cropping","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop3dLayer("Name","encoderDecoderSkipConnectionCrop2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,2,"Name","encoderDecoderSkipConnectionFeatureMerge2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],128,"Name","Decoder-Stage-2-Conv-1","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-2-BN-1")
    reluLayer("Name","Decoder-Stage-2-ReLU-1")
    convolution3dLayer([3 3 3],128,"Name","Decoder-Stage-2-Conv-2","Padding","same","WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-2-BN-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],128,"Name","Decoder-Stage-2-Conv-1_1","Padding","same","WeightsInitializer","he");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_7")
    reluLayer("Name","Decoder-Stage-2-ReLU-2")
    transposedConv3dLayer([2 2 2],64,"Name","Decoder-Stage-3-UpConv","BiasLearnRateFactor",2,"Stride",[2 2 2],"WeightsInitializer","he")
    reluLayer("Name","Decoder-Stage-3-UpReLU")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],64,"Name","conv3d","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    reluLayer("Name","relu")
    convolution3dLayer([1 1 1],64,"Name","conv3d_2","Padding","same")
    sigmoidLayer("Name","sigmoid")
    transposedConv3dLayer([1 1 1],64,"Name","transposed-conv3d","Cropping","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop3dLayer("Name","encoderDecoderSkipConnectionCrop1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","encoderDecoderSkipConnectionFeatureMerge1")
    convolution3dLayer([3 3 3],32,"Name","Decoder-Stage-3-Conv-1","Padding","same","WeightsInitializer","he")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = convolution3dLayer([1 1 1],32,"Name","Decoder-Stage-3-Conv-2_1","Padding","same","Stride",[1 1 2],"WeightsInitializer","he");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","Decoder-Stage-3-BN-1")
    reluLayer("Name","Decoder-Stage-3-ReLU-1")
    convolution3dLayer([3 3 3],32,"Name","Decoder-Stage-3-Conv-2","Padding","same","Stride",[1 1 2],"WeightsInitializer","he")
    batchNormalizationLayer("Name","Decoder-Stage-3-BN-2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_8")
    reluLayer("Name","Decoder-Stage-3-ReLU-2")
    convolution3dLayer([1 1 1],4,"Name","encoderDecoderFinalConvLayer")
    resize3dLayer("Name","resize3d-output-size_2","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[22 22 16])
   regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);



lgraph = connectLayers(lgraph,"resize3d-output-size_1","Encoder-Stage-1-Conv-1");
lgraph = connectLayers(lgraph,"resize3d-output-size_1","Encoder-Stage-1-Conv-2_1");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-Conv-2_1","addition_9/in1");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-BN-2","addition_9/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","conv3d_1");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-ReLU-2","Encoder-Stage-1-MaxPool");
lgraph = connectLayers(lgraph,"conv3d_1","addition/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-MaxPool","Encoder-Stage-2-Conv-1_1");
lgraph = connectLayers(lgraph,"Encoder-Stage-1-MaxPool","Encoder-Stage-2-Conv-1");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-Conv-1_1","addition_3/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-BN-2","addition_3/in1");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","Encoder-Stage-2-MaxPool");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-ReLU-2","conv3d_4");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-MaxPool","Encoder-Stage-3-Conv-1");
lgraph = connectLayers(lgraph,"Encoder-Stage-2-MaxPool","Encoder-Stage-3-Conv-2_1");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-BN-2","addition_4/in2");
lgraph = connectLayers(lgraph,"conv3d_4","addition_1/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-Conv-2_1","addition_4/in1");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","conv3d_7");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-ReLU-2","Encoder-Stage-3-DropOut");
lgraph = connectLayers(lgraph,"conv3d_7","addition_2/in2");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-MaxPool","LatentNetwork-Bridge-Conv-1");
lgraph = connectLayers(lgraph,"Encoder-Stage-3-MaxPool","LatentNetwork-Bridge-Conv-2_1");
lgraph = connectLayers(lgraph,"LatentNetworkBridge-BN-2","addition_5/in1");
lgraph = connectLayers(lgraph,"LatentNetwork-Bridge-Conv-2_1","addition_5/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","conv3d_6");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","multiplication_2/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","encoderDecoderSkipConnectionCrop3/ref");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-UpReLU","encoderDecoderSkipConnectionFeatureMerge3/in2");
lgraph = connectLayers(lgraph,"conv3d_6","addition_2/in1");
lgraph = connectLayers(lgraph,"transposed-conv3d_2","multiplication_2/in1");
lgraph = connectLayers(lgraph,"multiplication_2","encoderDecoderSkipConnectionCrop3/in");
lgraph = connectLayers(lgraph,"encoderDecoderSkipConnectionCrop3","encoderDecoderSkipConnectionFeatureMerge3/in1");
lgraph = connectLayers(lgraph,"encoderDecoderSkipConnectionFeatureMerge3","Decoder-Stage-1-Conv-1");
lgraph = connectLayers(lgraph,"encoderDecoderSkipConnectionFeatureMerge3","Decoder-Stage-1-Conv-2_1");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-BN-2","addition_6/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-1-Conv-2_1","addition_6/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU","conv3d_3");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU","multiplication_1/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU","encoderDecoderSkipConnectionCrop2/ref");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-UpReLU","encoderDecoderSkipConnectionFeatureMerge2/in2");
lgraph = connectLayers(lgraph,"conv3d_3","addition_1/in1");
lgraph = connectLayers(lgraph,"transposed-conv3d_1","multiplication_1/in1");
lgraph = connectLayers(lgraph,"multiplication_1","encoderDecoderSkipConnectionCrop2/in");
lgraph = connectLayers(lgraph,"encoderDecoderSkipConnectionCrop2","encoderDecoderSkipConnectionFeatureMerge2/in1");
lgraph = connectLayers(lgraph,"encoderDecoderSkipConnectionFeatureMerge2","Decoder-Stage-2-Conv-1");
lgraph = connectLayers(lgraph,"encoderDecoderSkipConnectionFeatureMerge2","Decoder-Stage-2-Conv-1_1");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-Conv-1_1","addition_7/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-2-BN-2","addition_7/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpReLU","conv3d");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpReLU","multiplication/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpReLU","encoderDecoderSkipConnectionCrop1/ref");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-UpReLU","encoderDecoderSkipConnectionFeatureMerge1/in2");
lgraph = connectLayers(lgraph,"conv3d","addition/in1");
lgraph = connectLayers(lgraph,"transposed-conv3d","multiplication/in1");
lgraph = connectLayers(lgraph,"multiplication","encoderDecoderSkipConnectionCrop1/in");
lgraph = connectLayers(lgraph,"encoderDecoderSkipConnectionCrop1","encoderDecoderSkipConnectionFeatureMerge1/in1");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-Conv-1","Decoder-Stage-3-Conv-2_1");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-Conv-1","Decoder-Stage-3-BN-1");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-Conv-2_1","addition_8/in2");
lgraph = connectLayers(lgraph,"Decoder-Stage-3-BN-2","addition_8/in1");
lgraph_MTAGU_Net=lgraph;
clear lgraph