import io, importlib, argparse
import torch, torch.nn as nn, torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--script_model', nargs=1, required=True, type=str,
					help='Script module for export')
parser.add_argument('--weight_file', nargs=1, required=False, type=str,
					help='Override weights by specific file')
parser.add_argument('--input_shape', nargs='+', required=True, type=int,
					help='Shape of example input for tracing')
parser.add_argument('--onnx_file', nargs=1, required=True, type=str,
					help='Output filename of ONNX model')
args = parser.parse_args()

def load_weights(model):
	f = open(args.weight_file[0], 'r')
	lines = f.read().splitlines()
	loaded_params = {}
	for line in lines:
		name, data = line.split(' ')
		tensor = torch.load(io.BytesIO(bytes.fromhex(data)))
		loaded_params[name] = tensor
	for name, param in model.named_parameters():
		param.requires_grad = False
		if name in loaded_params:
			param.copy_(loaded_params[name])


model = torch.jit.load(args.script_model[0])
model.train(False)
load_weights(model)

example_input = torch.randn(args.input_shape)
script_model = torch.jit.trace(model, example_input)
example_output = script_model.forward(example_input)

torch.onnx.export(model,
	example_input,
	args.onnx_file[0],
	verbose = True,
	output_names = ['output'],
	example_outputs = example_output
	)

print('Exported to ONNX')