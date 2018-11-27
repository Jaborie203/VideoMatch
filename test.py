from models.dilated_resnet import resnet101
model = resnet101(pretrained=True)
print(model)