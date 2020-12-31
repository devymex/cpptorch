import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def init_weight(m):
	if isinstance(m, nn.Conv1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.Conv3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose1d):
		init.normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose2d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.ConvTranspose3d):
		init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			init.normal_(m.bias.data)
	elif isinstance(m, nn.BatchNorm1d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm2d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm3d):
		init.normal_(m.weight.data, mean=1, std=0.02)
		init.constant_(m.bias.data, 0)
	elif isinstance(m, nn.Linear):
		init.xavier_normal_(m.weight.data)
		init.normal_(m.bias.data)
	elif isinstance(m, nn.LSTM):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.LSTMCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRU):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)
	elif isinstance(m, nn.GRUCell):
		for param in m.parameters():
			if len(param.shape) >= 2:
				init.orthogonal_(param.data)
			else:
				init.normal_(param.data)

def init_weights(model, url):
	loaded_params = load_state_dict_from_url(url, progress=True)
	model.apply(init_weight)
	for name, param in model.named_parameters():
		if name in loaded_params:
			if loaded_params[name].size() == param.data.size():
				param.data.copy_(loaded_params[name].data)
	return torch.jit.script(model)


class SimpleNet(nn.Module):
	def __init__(self):
		super(SimpleNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.fc1 = nn.Linear(320, 64)
		self.fc2 = nn.Linear(64, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = F.relu(self.fc2(x))
		return F.log_softmax(x, 1)

model = SimpleNet()
model.apply(init_weight)
model = torch.jit.script(model)
model.save("experiments/states/simplenet.pth")
