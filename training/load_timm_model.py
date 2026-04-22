import timm

import torch


if __name__ == "__main__":
    # Load the ResNet-18 model from timm
    model = timm.create_model('resnet18', pretrained=True, num_classes = 0)

    test_data = torch.randn(1,3,224, 224)

    model.eval()

    print(model.fc.in_features)

    with torch.no_grad():
        output = model(test_data)

    # Print the model architecture
    print(output.shape)