import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn as nn
import torch.nn.functional as F
import cv2

# mask = torch.load("/home/whl/workspace/cogvideo_edit/masks_resized.pt")
# plt.imshow(mask[5,0])
# plt.savefig("masks_resized.png")
# pdb.set_trace()

all_mask = torch.load("/home/whl/workspace/cogvideo_edit/metrics_mask/fox_all_category.pt") #[9,480,720]

output = F.max_pool2d(all_mask.float(), kernel_size=(8,8), stride=(8,8))
output= output.reshape(3, 3, 60, 90)
output = output.max(dim=1)[0]
#output = output[:,1]
output = output.bool()
kernel = np.ones((3, 3), np.uint8) 
for i in range(output.shape[0]):
    mask = output[i].numpy()*255
    mask = mask.astype(np.uint8)
    mask_new = cv2.dilate(mask, kernel, iterations=1)
    plt.imshow(mask_new, cmap = "hot")
    plt.savefig(f"/home/whl/workspace/cogvideo_edit/Detic/demo_test/fox_all_category_resize_3_dilate{i}.png")
pdb.set_trace()

for i in range(9):
    mask = all_mask[i].numpy().astype(int)
    plt.imshow(mask, cmap = "gray")
    plt.savefig(f"/home/whl/workspace/cogvideo_edit/metrics_mask_visualize/swan_{i}.png")
    
pdb.set_trace()
    
avg_pool = nn.AvgPool2d(kernel_size=(17, 25), stride=(17, 25))
#max_pool = nn.MaxPool2d(kernel_size=(17, 25), stride=(17, 25))
output = avg_pool(all_mask[0].unsqueeze(0).unsqueeze(0).float())
#output = max_pool(all_mask[0].unsqueeze(0).unsqueeze(0).float())
plt.imshow(output.squeeze())
plt.savefig("/home/whl/workspace/cogvideo_edit/metrics_mask_visualize/test.png")
pdb.set_trace()
    
# class_name = torch.load("/home/whl/workspace/cogvideo_edit/Detic/demo_class_name.pt")
# pred_masks = torch.load("/home/whl/workspace/cogvideo_edit/Detic/demo_pred_masks.pt")
# pred_class = torch.tensor([ 792,   93,  950,  555, 1177,  980,  947,  539,  594])

# for i in range(len(pred_class)):
#     if class_name[pred_class[i]] == "person":
#         mask=pred_masks[i].numpy().astype(int)
#         plt.imshow(mask, cmap = "gray")
#         plt.savefig("/home/whl/workspace/cogvideo_edit/Detic/demo_mask.png")