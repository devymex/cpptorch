import sys, os.path as path, functools as ft        # basic packages
import urllib.request as url_req, numpy as np, cv2  # extension packages
import torch, torchvision.ops as ops                # pytorch packages

data_path = '.' if len(sys.argv) < 2 else sys.argv[1]
image_file_url = 'https://github.com/pjreddie/darknet/raw/master/data/dog.jpg'
image_filename = path.join(data_path, 'dog.jpg')
torchscript_file = path.join(data_path, 'yolov3.pth')

num_cls    = 80
conf_thres = 0.24
nms_thres  = 0.45
input_size = (416, 416)

def nms_comp(k):
    def op(box_a, box_b):
        if box_a[5 + k] < box_b[5 + k]:   return 1
        elif box_a[5 + k] > box_b[5 + k]: return -1
        else:                             return 0
    return op

def nms(all_boxes):
    for k in range(0, num_cls):
        all_boxes = sorted(all_boxes, key=ft.cmp_to_key(nms_comp(k)))
        all_boxes = torch.cat(all_boxes).reshape(-1, 5 + num_cls)
        for i in range(0, all_boxes.shape[0] - 1):
            if all_boxes[i][5+k] != 0:
                boxes_i = all_boxes[i:i+1,:4]
                IoUs = ops.box_iou(boxes_i, all_boxes[i+1:,:4])
                for j, iou in enumerate(IoUs[0]):
                    if iou > nms_thres:
                        all_boxes[i+1+j,5+k] = 0
    return all_boxes

torch.set_printoptions(sci_mode=False)

if not path.exists(image_filename):
    print('Downloading test image ...')
    url_req.urlretrieve(image_file_url, image_filename)

model = torch.jit.load(torchscript_file)
model.eval()

# Load image data and do preprocessing
img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
data = cv2.cvtColor(cv2.resize(img, input_size), cv2.COLOR_RGB2BGR)
data = torch.tensor(data) / 255.0
data = data.permute(2, 0, 1).reshape(1, 3, input_size[1], input_size[0])

model_outputs = model.forward(data)
reshaped_outputs = []
for output in model_outputs:
    output = output.detach().reshape(1, -1, output.shape[-1])
    reshaped_outputs.append(output[:,:,output.shape[-1] - 5 - num_cls:])
predict = torch.cat(reshaped_outputs, dim=1)
predict[:,:,5:] *= predict[:,:,4].unsqueeze(2) # prob *= conf
predict[:,:,5:] *= torch.gt(predict[:,:,5:], conf_thres) # low conf are masked out 

all_boxes = []
for boxes in predict[0]:
	conf = boxes[4]
	if conf > conf_thres:
		all_boxes.append(boxes.view(1, -1))

best_boxes = []
if len(all_boxes) > 0:
	all_boxes = nms(torch.cat(all_boxes))
	cls_ids = torch.argmax(all_boxes[:, 5:], dim=1)
	for box_idx, best_cls_id in enumerate(cls_ids):
		if all_boxes[box_idx, 5 + best_cls_id] > conf_thres:
			box = torch.cat((all_boxes[box_idx, :4],
				torch.tensor([np.float32(best_cls_id)])))
			best_boxes.append(box)

img_size = (img.shape[1], img.shape[0])
for box in best_boxes: # Draw and save precition results
    pt1 = (int(box[0] * img_size[0]), int(box[1] * img_size[1]))
    pt2 = (int(box[2] * img_size[0]), int(box[3] * img_size[1]))
    cv2.rectangle(img, pt1, pt2, (0, 255, 0))
cv2.imwrite(path.join(data_path, 'predict.png'), img)