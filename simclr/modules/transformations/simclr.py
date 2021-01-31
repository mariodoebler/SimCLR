import torchvision
import torch
import torch.nn

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

class TransformsSimCLRAtari:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, width, height):
        s = 1
        # color_jitter = torchvision.transforms.ColorJitter(
        #     0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        # )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                # size: FINALE size
                torchvision.transforms.RandomResizedCrop(size=(height, width), scale=(0.9, 0.95)),
                # torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                # torchvision.transforms.RandomApply([color_jitter], p=0.8),
                # torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(height, width)),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        assert x.shape[0] == 4
        # framestack_im = transforms.ToPILImage()(framestack) #.convert("RGB")
        # sample1=train_transform(transforms.ToPILImage()(x[0, :, :]).convert('L'))  # L-mode: 8bit, 1 channel
        # sample2=train_transform(transforms.ToPILImage()(x[0, :, :]).convert('L'))  # L-mode: 8bit, 1 channel
        # for i in arange(1,4):
        #     torch.cat

        sample1 =  self.random_convolution(self.train_transform(x))
        sample2 = self.random_convolution(self.train_transform(x))
        # sample1 = self.train_transform(x)
        # sample2 = self.train_transform(x)

        # im_sample1 = sample1.squeeze()
        # im1 = im_sample1[0, :, :]
        # im_sample2 = sample2.squeeze()
        # im2 = im_sample2[0, :, :]
        # for i in range(1, 4):
        #     # 3, frameestack, height, width
        #     im1 = torch.cat((im1, im_sample1[i, :, :]), axis=-1)  # via the last axis --> width
        #     im2 = torch.cat((im2, im_sample2[i, :, :]), axis=-1)  # via the last axis --> width
        # im1 = torchvision.transforms.ToPILImage()(im1)
        # im2 = torchvision.transforms.ToPILImage()(im2)
        # im1.save(f'/home/cathrin/MA/datadump/simclr/augs1.png')
        # im2.save(f'/home/cathrin/MA/datadump/simclr/augs2.png')
        return sample1, sample2

    def random_convolution(self, imgs):
        '''
        random covolution in "network randomization"

        (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor
        '''
        assert len(imgs.shape) == 3
        assert imgs.shape[0] == 4
        _device = imgs.device

        img_h, img_w = imgs.shape[-2], imgs.shape[-1]
        num_stack_channel = imgs.shape[-3]
        # num_batch = imgs.shape[0]
        # num_trans = num_batch
        # batch_size = int(num_batch / num_trans)
        batch_size = 1

        # initialize random covolution
        rand_conv = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(_device).requires_grad_(False)

        # for trans_index in range(num_trans):
        torch.nn.init.xavier_normal_(rand_conv.weight.data)
        # temp_imgs = imgs[trans_index * batch_size:(trans_index + 1) * batch_size]
        for i in range(num_stack_channel):
            temp_imgs = imgs[i, :, :].unsqueeze(0).unsqueeze(0)#[1:(trans_index + 1) * batch_size]

            # temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w)  # (batch x stack, channel, h, w)
            rand_out = rand_conv(temp_imgs)
            if i == 0:
                total_out = rand_out
            else:
                total_out = torch.cat((total_out, rand_out), 0)
        total_out = total_out.reshape(num_stack_channel, img_h, img_w)
        return total_out.squeeze()