#!/bin/bash

python3 scripts/script_model \
	--model_name simple_model \
	--script_model experiments/states/simplenet_s.pth

python3 scripts/script_model \
	--model_name simple_model \
	--trace_input 1 1 28 28 \
	--script_model experiments/states/simplenet_t.pth

python3 scripts/export_onnx.py \
	--script_model experiments/states/simplenet_t.pth \
	--onnx_file experiments/states/simplenet_t.onnx \
	--input_shape 1 1 28 28 \
	--weight_file experiments/states/

/usr/local/deployment_tools/model_optimizer/mo.py \
    --data_type FP32 \
    --model_name model \
    --output_dir experiments/states/vino \
    --input_model experiments/states/simplenet_t.onnx \
    --input image \
	--input_shape [1,1,28,28] \
	--output predict
