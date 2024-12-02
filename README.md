# Project report

In this project I am going to recreate the neural network from the article called
["LightSleepNet: Design of a Personalized Portable Sleep Staging System Based on Single-Channel EEG"](https://arxiv.org/abs/2401.13194v1)
and evaluate its performance on public sleep dataset.

In the original article the model was evaluated using ["Sleep-EDF"](https://physionet.org/content/sleep-edf/1.0.0/) dataset and only one (Fpz-Cz) EEG channel.

I'm going to perform my experiments on another dataset called ["ISRUC-Sleep"](https://sleeptight.isr.uc.pt) on 6 different EEG channels:
F3-A2, C3-A2, O1-A2, F4-A1, C4-A1 and O2-A1.

## What have I already done

I have preprocessed the data from ISRUC-Sleep dataset by renaming channels and
adding sleep stage annotations to 30-second epochs. I've made 3 types of annotations:
- annotations done by 1st annotator (for now I am experimenting only with this one);
- annotations done by 2nd annotator;
- annotations on epochs where both 1st and 2nd annotators agree on sleep stage.

I've made a neural network that follows the architecture described in the article.
It still misses some features, but it's already working and producing decent result.

NN features that I've already done:
- two residual convolution blocks;
- channel shuffle after each convolutional layer;
- weighted cross entropy loss based on gradient density.

## What else do I need to do
- implement adaptive batch normalization after each convolution (for now replaced with regular BatchNorm1d);
- implement global average pooling (for now replaced with AdaptiveAvgPool1d);
- perform experiments on different train-test splits (for now I use 1st recording as test and all other recordings as train data);
- perform experiments on different EEG channels (for now I use only F3-A2).
