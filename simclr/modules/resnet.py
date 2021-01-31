import torchvision
from torch.nn import Conv2d

def get_resnet_atari(framestack=4):
    resnet = torchvision.models.resnet18(pretrained=False)
    if framestack == 4:
        # keep all settings EXCEPT in_channels --> now 4 instead of just 3
        resnet.conv1 = Conv2d(in_channels=4, out_channels=64, kernel_size=(7, 7), stride=(2,2), padding=(3,3), bias=False)
        return resnet
    elif framestack == 3:
        return resnet
    else:
        raise NotImplementedError(f"No Resnet available for framestack {framestack}")

def get_resnet(name, pretrained=False, framestack=4):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    if framestack == 4:
        return resnets[name]
    elif framestack == 3:
        return resnets[name]
