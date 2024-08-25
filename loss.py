import torch
from torch import nn
from torchvision.models.vgg import vgg16, vgg19
from torchvision.models import VGG16_Weights, VGG19_Weights


class GeneratorLoss(nn.Module):
    def __init__(self, configs):
        super(GeneratorLoss, self).__init__()
        if configs.perceptual_loss == "vgg16":
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        elif configs.perceptual_loss == "vgg19":
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.adversarial_criterion = nn.BCEWithLogitsLoss()

    def forward(self, out_labels, target_labels, out_images, target_images, bound=None):
        # Adversarial Loss
        valid = torch.ones((out_labels.shape[0], 1)).to(out_labels.device)
        adversarial_loss = self.adversarial_criterion(out_labels - target_labels.mean(), valid)
        # Perception Loss
        if bound is None:
            perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        else:
            first_loss = self.mse_loss(self.loss_network(out_images[:, :3, :, :]), self.loss_network(target_images[:, :3, :, :]))
            last_loss = self.mse_loss(self.loss_network(out_images[:, -3:, :, :]), self.loss_network(target_images[:, -3:, :, :]))
            perception_loss = (first_loss + last_loss) / 2
        # Image Loss
        if bound is None:
            image_loss = self.mse_loss(out_images, target_images)
        else:
            image_loss = self.mse_loss(out_images[bound == 1], target_images[bound == 1])
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x, bound=None):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
