import sys, io, os.path as path                 # basic packages
import urllib.request as url_req, numpy as np   # extension packages
import torch, torch.nn as nn                    # pytorch packages

data_path = '.' if len(sys.argv) < 2 else sys.argv[1]
weight_file_url = 'https://pjreddie.com/media/files/yolov3.weights'
weight_filename = path.join(data_path, 'yolov3.weights')
torchscript_file = path.join(data_path, 'yolov3.pth')
torch.set_printoptions(sci_mode=False)

num_cls    = 80
input_size = (416, 416)
anchors    = [(10,13),  (16,30),   (33,23),    # 0, 1, 2
              (30,61),  (62,45),   (59,119),   # 3, 4, 5
              (116,90), (156,198), (373,326)]  # 6, 7, 8
masks      = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] # yolo1, yolo2, yolo3

class YoloConv(nn.Module):
    def __init__(self, bn, c, n, size, stride, pad, leaky_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(c, n, (size, size), (stride, stride),
            (pad, pad), bias=not bn)
        if bn:
            self.bn = nn.BatchNorm2d(num_features=n, affine=True)
        if leaky_relu:
            self.active = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        if hasattr(self, 'active'):
            x = self.active(x)
        return x

class YoloResBlock(nn.Module):
    def __init__(self, c, n, res):
        super().__init__()
        self.res = res
        self.yolo_conv_1 = YoloConv(True, c, n // 2, 1, 1, 0)
        self.yolo_conv_2 = YoloConv(True, n // 2, n, 3, 1, 1)

    def forward(self, x):
        y = self.yolo_conv_1(x)
        y = self.yolo_conv_2(y)
        if self.res:
            y = x + y
        return y

class Yolov3Backbone(nn.Module):
    def __init__(self, num_vals):
        super().__init__()
        seg1_mods  = [YoloConv(True, 3, 32, 3, 1, 1), # input: data
                      YoloConv(True, 32, 64, 3, 2, 1),
                      YoloResBlock(64, 64, True),
                      YoloConv(True, 64, 128, 3, 2, 1),
                      YoloResBlock(128, 128, True),
                      YoloResBlock(128, 128, True),
                      YoloConv(True, 128, 256, 3, 2, 1)]
        for i in range(8):
            seg1_mods.append(YoloResBlock(256, 256, True))
        seg2_mods  =  [YoloConv(True, 256, 512, 3, 2, 1)] # input: seg1_out
        for i in range(8):
            seg2_mods.append(YoloResBlock(512, 512, True))
        seg3_mods =  [YoloConv(True, 512, 1024, 3, 2, 1), # input: seg2_out
                      YoloResBlock(1024, 1024, True),
                      YoloResBlock(1024, 1024, True),
                      YoloResBlock(1024, 1024, True),
                      YoloResBlock(1024, 1024, True),
                      YoloResBlock(1024, 1024, False),
                      YoloResBlock(1024, 1024, False),
                      YoloConv(True, 1024, 512, 1, 1, 0)]
        yolo1_mods = [YoloConv(True, 512, 1024, 3, 1, 1), # input: seg3_out
                      YoloConv(False, 1024, num_vals, 1, 1, 0, False)]
        seg4_mods  = [YoloConv(True, 512, 256, 1, 1, 0), # input: seg2_out
                      nn.Upsample(scale_factor=2)]
        seg5_mods  = [YoloResBlock(768, 512, False), # input: seg4_out, seg2_out
                      YoloResBlock(512, 512, False),
                      YoloConv(True, 512, 256, 1, 1, 0)]
        yolo2_mods = [YoloConv(True, 256, 512, 3, 1, 1),# yolo2 input: seg5_out
                      YoloConv(False, 512, num_vals, 1, 1, 0, False)]
        seg6_mods  = [YoloConv(True, 256, 128, 1, 1, 0), # input: seg5_out
                      nn.Upsample(scale_factor=2)]
        yolo3_mods = [YoloResBlock(384, 256, False), # input: seg6, seg1_out
                      YoloResBlock(256, 256, False),
                      YoloResBlock(256, 256, False),
                      YoloConv(False, 256, num_vals, 1, 1, 0, False)]

        # DO NOT REORDER FOLLOWING SEQUENTIALS
        # the following order is consistent with the cfg file `yolov3-voc.cfg`
        #     and the weights file `yolov3.weights`
        self.seg1  = nn.Sequential(*seg1_mods)
        self.seg2  = nn.Sequential(*seg2_mods)
        self.seg3  = nn.Sequential(*seg3_mods)
        self.yolo1 = nn.Sequential(*yolo1_mods)
        self.seg4  = nn.Sequential(*seg4_mods)
        self.seg5  = nn.Sequential(*seg5_mods)
        self.yolo2 = nn.Sequential(*yolo2_mods)
        self.seg6  = nn.Sequential(*seg6_mods)
        self.yolo3 = nn.Sequential(*yolo3_mods)

    def forward(self, x):
        seg1_out  = self.seg1.forward(x)
        seg2_out  = self.seg2.forward(seg1_out)
        seg3_out  = self.seg3.forward(seg2_out)
        yolo1_out = self.yolo1.forward(seg3_out)
        seg4_out  = self.seg4.forward(seg3_out)
        cat42     = torch.cat((seg4_out, seg2_out), 1)
        seg5_out  = self.seg5.forward(cat42)
        yolo2_out = self.yolo2.forward(seg5_out)
        seg6_out  = self.seg6.forward(seg5_out)
        cat61     = torch.cat((seg6_out, seg1_out), 1)
        yolo3_out = self.yolo3.forward(cat61)
        return yolo1_out, yolo2_out, yolo3_out

class YoloLayer(nn.Module):
    def __init__(self, anchors, masks):
        super().__init__()
        self.anchors = anchors
        self.masks = masks
        self.register_buffer('scale_w', torch.zeros(1))
        self.register_buffer('scale_h', torch.zeros(1))
        self.register_buffer('anc_off_c', torch.zeros(1))
        self.register_buffer('anc_off_r', torch.zeros(1))

    def forward(self, input):
        # input: batch*anc*vals*h*w -> vals*batch*anc*h*w
        input = input.reshape(self.calc_shape).permute(2, 0, 1, 3, 4)
        box_cx = torch.sigmoid(input[0:1]) # sigmoid(px)
        box_cy = torch.sigmoid(input[1:2]) # sigmoid(py)
        box_w = input[2:3]
        box_h = input[3:4]
        conf_probs = torch.sigmoid(input[4:]) # objectness and probabilities

        box_ws = torch.exp(box_w) * self.scale_w # exp(pw) * anc_w / img_w
        box_hs = torch.exp(box_h) * self.scale_h # exp(ph) * anc_h / img_h
        box_x1 = (box_cx + self.anc_off_c) / self.anc_cols - box_ws / 2
        box_y1 = (box_cy + self.anc_off_r) / self.anc_rows - box_hs / 2
        box_x2 = box_x1 + box_ws # x2 = x1 + w
        box_y2 = box_y1 + box_hs # y2 = y1 + h

        output = torch.cat((box_cx, box_cy, box_w, box_h,
                box_x1, box_y1, box_x2, box_y2, conf_probs))
        # output: vals*batch*anc*h*w -> batch*anc*h*w*vals
        return output.permute(1, 2, 3, 4, 0)

    @torch.no_grad()
    def resize(self, input, image_size):
        num_ancs = len(self.masks)
        num_vals = int(input.shape[1] // num_ancs)
        self.anc_rows = int(input.shape[2])
        self.anc_cols = int(input.shape[3])
        self.calc_shape = [-1, num_ancs, num_vals, self.anc_rows, self.anc_cols]

        scale_w = torch.tensor([self.anchors[m][0] for m in self.masks],
            device=self.scale_w.device)
        scale_h = torch.tensor([self.anchors[m][1] for m in self.masks],
            device=self.scale_h.device)
        self.scale_w = scale_w.reshape([1, 1, num_ancs, 1, 1]) / image_size[0]
        self.scale_h = scale_h.reshape([1, 1, num_ancs, 1, 1]) / image_size[1]

        off_c = torch.arange(self.anc_cols, dtype=torch.float32,
            device=self.anc_off_c.device)
        off_r = torch.arange(self.anc_rows, dtype=torch.float32,
            device=self.anc_off_r.device)
        self.anc_off_c = off_c.reshape([1, 1, 1, 1, self.anc_cols])
        self.anc_off_r = off_r.reshape([1, 1, 1, self.anc_rows, 1])

class Yolov3Model(nn.Module):
    def __init__(self, num_cls, image_size, anchors, masks):
        super().__init__()
        image_size = torch.tensor([image_size[0], image_size[1]])
        self.register_buffer('image_size', image_size)
        self.backbone = Yolov3Backbone(len(masks) * (5 + num_cls))

        example_input = torch.zeros([1, 3, image_size[1], image_size[0]])
        example_output = self.backbone.forward(example_input)
        yolo_layers = [YoloLayer(anchors, m) for m in masks]
        for i, input in enumerate(example_output):
            yolo_layers[i].resize(input, image_size)
        self.yolo_mods = nn.ModuleList(yolo_layers)

    def forward(self, input):
        outputs = self.backbone(input)
        self.image_size[0].fill_(input.shape[3])
        self.image_size[1].fill_(input.shape[2])
        if self.training:
            self.yolo_mods[0].resize(outputs[0], self.image_size)
            self.yolo_mods[1].resize(outputs[1], self.image_size)
            self.yolo_mods[2].resize(outputs[2], self.image_size)
        output0 = self.yolo_mods[0](outputs[0])
        output1 = self.yolo_mods[1](outputs[1])
        output2 = self.yolo_mods[2](outputs[2])
        return (output0, output1, output2)

def load_param(tensor, bin_stream):
    data = bin_stream.read(tensor.numel() * 4)
    if len(data) == tensor.numel() * 4:
        float_ary = np.frombuffer(data, dtype='<f4')
        tensor.copy_(torch.tensor(float_ary.copy()).reshape_as(tensor))

@torch.no_grad()
def load_darknet_params_file(filename, module):
    bin_stream = open(filename, 'rb')
    bin_stream.read(20) # skip the file header
    for module in model.modules():
        if isinstance(module, YoloConv):
            if hasattr(module, 'bn'):
                load_param(module.bn.bias, bin_stream)
                load_param(module.bn.weight, bin_stream)
                load_param(module.bn.running_mean, bin_stream)
                load_param(module.bn.running_var, bin_stream)
            else:
                load_param(module.conv.bias, bin_stream)
            load_param(module.conv.weight, bin_stream) # all convs have weight


# Create model and do inference
model = Yolov3Model(num_cls, input_size, anchors, masks)
model.train()

# Download image and weight files if they do not exist
if not path.exists(weight_filename):
    print('Downloading weights file ...')
    url_req.urlretrieve(weight_file_url, weight_filename)
load_darknet_params_file(weight_filename, model)
script_model = torch.jit.script(model)
torch.jit.save(script_model, torchscript_file)

# Testing for trace
model.eval()
sample_data = torch.zeros(1, 3, input_size[1], input_size[0])
trace_model = torch.jit.trace(model, sample_data)
