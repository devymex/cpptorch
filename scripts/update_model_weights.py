import io, argparse, torch

parser = argparse.ArgumentParser()
parser.add_argument('--script_model', nargs=1, required=True, type=str,
					help='Script module for export')
parser.add_argument('--weight_file', nargs=1, required=False, type=str,
					help='Override weights by specific file')
parser.add_argument('--output_file', nargs=1, required=False, type=str,
					help='Override weights by specific file')
args = parser.parse_args()

def load_weights(model):
	f = open(args.weight_file[0], 'r')
	lines = f.read().splitlines()
	loaded_params = {}
	for line in lines:
		name, data = line.split(' ')
		tensor = torch.load(io.BytesIO(bytes.fromhex(data)))
		loaded_params[name] = tensor.cpu()
	for name, param in model.named_parameters():
		param.requires_grad = False
		if name in loaded_params:
			param.copy_(loaded_params[name])

model = torch.jit.load(args.script_model[0])
model.cpu()
model.train(False)
if args.weight_file is not None:
	load_weights(model)

torch.jit.save(model, args.output_file[0])