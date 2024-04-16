import timm  # noqa
import torchvision.models as models  # noqa
import patchcore.vision_transformer as vits
import torch
import unicom

_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnet152": "models.resnet152(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "dino_resnet50": "torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')",
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}


def load(name):
    if name == 'relicv2_50':
        model = models.resnet50()
        model.load_state_dict(torch.load('/home/maometus/Documents/projects/relic/relicv2_checkpoint_R50_1x.pkl'))
        model.eval()
        return

    if name == 'relicv2_101':
        model = models.resnet101()
        model.load_state_dict(torch.load('/home/maometus/Documents/projects/relic/relicv2_checkpoint_R101_1x.pkl'))
        model.eval()
        return

    if name == 'relicv2_152':
        model = models.resnet152()
        model.load_state_dict(torch.load('/home/maometus/Documents/projects/relic/relicv2_checkpoint_R152_1x.pkl'))
        model.eval()
        return

    if name == 'relicv2_200':
        model = models.resnet200()
        model.load_state_dict(torch.load('/home/maometus/Documents/projects/relic/relicv2_checkpoint_R200_1x.pkl'))
        model.eval()
        return

    if name == 'custom_iad':
        checkpoint = torch.load('/home/maometus/Documents/projects/checkpoint.pth')
        state_dict = {}
        for _, (k, v) in enumerate(checkpoint['student'].items()):
            if k.startswith('module.backbone.'):
                state_dict[k.replace('module.backbone.', '')] = v
        model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.load_state_dict(state_dict)
        model.eval()
        return model

    if name == 'custom_iad_vit_base':
        checkpoint = torch.load('/home/maometus/Documents/projects/custom_iad_vit_base_ep100.pth')
        state_dict = {}
        for _, (k, v) in enumerate(checkpoint['student'].items()):
            if k.startswith('module.backbone.'):
                state_dict[k.replace('module.backbone.', '')] = v
        model = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    if name == 'unicom_b16':
        return unicom.load('ViT-B/16')[0]

    return eval(_BACKBONES[name])
