import torch
import torchvision.models as models
from torchvision.io import read_image

# model = models.vgg16(weights=None)
# pre = torch.load('./backbone/vgg16-397923af.pth')
# model.load_state_dict(pre)





img = read_image("./test/0041.png")

# Step 1: Initialize model with the best available weights
weights = None
model = models.vgg16(weights=None)
pre = torch.load('./backbone/vgg16-397923af.pth')
model.load_state_dict(pre)
model.eval()
print(model)
weights = models.VGG16_Weights.IMAGENET1K_V1

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)#因为是一张图片，而大部分都是一个批次输入的故要添加一个维度

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)#squeeze删除尺度为1的维度，squeeze(0)，若第0维尺度为1则删掉否则不变
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")