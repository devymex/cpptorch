import io, sys, struct, torch

def load_buffer(stream):
	bin_size = stream.read(8)
	if len(bin_size) == 0:
		return None
	size = struct.unpack('<Q', bin_size)[0]
	return stream.read(size)

def load_named_tensor(stream):
	name = load_buffer(stream)
	buf = load_buffer(stream)
	if name is None or buf is None:
		return None
	name = name.decode()
	buf = torch.load(io.BytesIO(buf))
	return name, buf

model = torch.jit.load(sys.argv[1])
model.cpu()
model.train(False)

loaded_params = {}
bin_file = open(sys.argv[2], 'rb')
while True:
	named_tensor = load_named_tensor(bin_file)
	if named_tensor is None:
		break
	loaded_params[named_tensor[0]] = named_tensor[1]

for name, param in model.named_parameters():
	param.requires_grad = False
	if name in loaded_params:
		param.copy_(loaded_params[name])
		print('Loaded', name)

for name, buf in model.named_buffers():
	if name in loaded_params:
		buf.copy_(loaded_params[name])
		print('Loaded', name)

torch.jit.save(model, sys.argv[3])