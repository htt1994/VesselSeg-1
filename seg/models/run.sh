#!/bin/bash

#Pretrained model path
MODEL_PATH=baseline-resnet50dilated-ppm_deepsup
RESULT_PATH=./

ENCODER=$MODEL_PATH/encoder_epoch_20.pth
DECODER=$MODEL_PATH/decoder_epoch_20.pth

# Download model weights and image
if [ ! -e $MODEL_PATH ]; then
  mkdir $MODEL_PATH
fi
if [ ! -e $ENCODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
fi
if [ ! -e $DECODER ]; then
  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
fi


python3 -u train.py \
  --gpus 0
  --model_path $MODEL_PATH \
  --arch_encoder resnet50dilated \
  --arch_decoder ppm_deepsup \
  --fc_dim 2048 \
  --result $RESULT_PATH

python3 -u
