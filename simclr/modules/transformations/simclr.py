import numpy as np
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

class BatchwiseTransformsSimCLRAtari:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, input_sample, random_cropping, batch_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.padding = torch.nn.ReplicationPad2d((12, 12, 16, 16)).to(self.device)
        self.random_cropping = random_cropping
        self.batch_size = batch_size
        self.num_channels, self.img_height, self.img_width = input_sample.shape
        self.rand_conv = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(self.device).requires_grad_(False)
        # if self.random_cropping:
        #     self.train_transform = torchvision.transforms.Compose(
        #         [
        #             torchvision.transforms.ToPILImage(),
        #             # torchvision.transforms.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0)),
        #             torchvision.transforms.RandomResizedCrop(size=(height, width), scale=(0.85, 1.0)),
        #             # torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
        #             # torchvision.transforms.RandomApply([color_jitter], p=0.8),
        #             # torchvision.transforms.RandomGrayscale(p=0.2),
        #             torchvision.transforms.ToTensor(),
        #         ]
        #     )
        # else:
        #     self.train_transform = torchvision.transforms.Compose(
        #         [
        #             torchvision.transforms.ToPILImage(),
        #             # torchvision.transforms.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0)),
        #             # torchvision.transforms.RandomResizedCrop(size=(height, width), scale=(0.85, 1.0)),
        #             # torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
        #             # torchvision.transforms.RandomApply([color_jitter], p=0.8),
        #             # torchvision.transforms.RandomGrayscale(p=0.2),
        #             torchvision.transforms.ToTensor(),
        #         ]
        #     )
        #
        # self.test_transform = torchvision.transforms.Compose(
        #     [
        #         torchvision.transforms.Resize(size=(height, width)),
        #         torchvision.transforms.ToTensor(),
        #     ]
        # )

    def __call__(self, x):
        x = x.to(self.device)
        assert x.shape[1] == 4

        if torch.max(x) > 1.1:
            x = x/255.

        if self.random_cropping:
            sample1, sample2 = self.first_pad_then_random_crop(x)
        else:
            sample1 = x
            sample2 = x.clone()

        sample1, sample2 = self.adjust_brightness(sample1, sample2)
        sample1 = self.random_convolution(sample1)
        sample2 = self.random_convolution(sample2)

        mean1 = torch.mean(sample1)
        mean2 = torch.mean(sample2)
        if mean1 < 0.2:
            sample1 += 0.5
        if mean2 < 0.2:
            sample2 += 0.5
        if mean1 > 0.8:
            sample1 -= 0.5
        if mean2 > 0.8:
            sample2 -= 0.5

        sample1 = torch.clamp(sample1, 0, 1.)
        sample2 = torch.clamp(sample2, 0, 1.)

### for plotting / debugging
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
        # random_int = np.random.randint(20)
        # im1.save(f'/home/cathrin/MA/datadump/simclr/'+str(random_int) + '_1.png')
        # im2.save(f'/home/cathrin/MA/datadump/simclr/'+str(random_int) + '_2.png')
        return sample1, sample2

    def first_pad_then_random_crop(self, img_stack):
        # img_stack_dim4 = img_stack.unsqueeze(0) # now dim 4
        # assert len(img_stack_dim4.shape) == 4, "4 dimensions needed for replicationpad2d (implementation constraint)"
        # obs_padded = self.padding(img_stack_dim4).squeeze(0) # dim 3 again
        obs_padded = self.padding(img_stack) # .squeeze(0) # dim 3 again
        _, c, h, w = obs_padded.shape
        # generate random int between 0 and (w-160+1)-1  !
        w_shift = torch.randint(w - 160 + 1, size=(self.batch_size, 2), device=self.device)
        h_shift = torch.randint(h - 210 + 1, size=(self.batch_size, 2), device=self.device)
        img_stack1 = obs_padded[:, :, h_shift[:, 0]:(h_shift[:, 0]+210), w_shift[:, 0]:(w_shift[:, 0]+160)]
        img_stack2 = obs_padded[:, :, h_shift[:, 1]:(h_shift[:, 1]+210), w_shift[:, 1]:(w_shift[:, 1]+160)]
        assert img_stack1.shape[1:] == img_stack2.shape[1:] == (4, 210, 160), f"shape is {img_stack2.shape}, input shape was: {obs_padded.shape} w: {w_shift[1]}, h: {h_shift[1]}"
        return img_stack1, img_stack2

    def random_convolution(self, imgs):
        '''
        random covolution in "network randomization"

        (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor

        adjusted to just pass one channel per forward-pass. finally stack the outputs together
        '''
        # assert len(imgs.shape) == 3
        # assert imgs.shape[1] == 4


        # initialize random covolution

        torch.nn.init.xavier_normal_(self.rand_conv.weight.data)
        for i in range(num_stack_channel):
            # temp_imgs = imgs[i, :, :].unsqueeze(0).unsqueeze(0)
            temp_imgs = imgs[:, i, :, :].unsqueeze(1)

            # pass EACH single frame of the observation through the conv-layer
            rand_out = self.rand_conv(temp_imgs)
            if i == 0:
                total_out = rand_out
            else:
                total_out = torch.cat((total_out, rand_out), 0)
        total_out = total_out.reshape(self.batch_size, self.num_channels, self.img_height, self.img_width)
        return total_out.squeeze()

    def adjust_brightness(self, x1, x2):
        brightness_fct = torch.cuda.FloatTensor(self.batch_size, 2).uniform_(0.92, 1.08)
        return torchvision.transforms.functional.adjust_brightness(x1, brightness_factor=brightness_fct[0]).to(self.device), torchvision.transforms.functional.adjust_brightness(x2, brightness_factor=brightness_fct[1]).to(self.device)


class TransformsSimCLRAtari:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, width, height, random_cropping):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.padding = torch.nn.ReplicationPad2d((12, 12, 16, 16)).to(self.device)
        self.random_cropping = random_cropping
        self.rand_conv = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(self.device).requires_grad_(False)
        # if self.random_cropping:
        #     self.train_transform = torchvision.transforms.Compose(
        #         [
        #             torchvision.transforms.ToPILImage(),
        #             # torchvision.transforms.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0)),
        #             torchvision.transforms.RandomResizedCrop(size=(height, width), scale=(0.85, 1.0)),
        #             # torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
        #             # torchvision.transforms.RandomApply([color_jitter], p=0.8),
        #             # torchvision.transforms.RandomGrayscale(p=0.2),
        #             torchvision.transforms.ToTensor(),
        #         ]
        #     )
        # else:
        #     self.train_transform = torchvision.transforms.Compose(
        #         [
        #             torchvision.transforms.ToPILImage(),
        #             # torchvision.transforms.RandomResizedCrop(size=(height, width), scale=(0.8, 1.0)),
        #             # torchvision.transforms.RandomResizedCrop(size=(height, width), scale=(0.85, 1.0)),
        #             # torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
        #             # torchvision.transforms.RandomApply([color_jitter], p=0.8),
        #             # torchvision.transforms.RandomGrayscale(p=0.2),
        #             torchvision.transforms.ToTensor(),
        #         ]
        #     )
        #
        # self.test_transform = torchvision.transforms.Compose(
        #     [
        #         torchvision.transforms.Resize(size=(height, width)),
        #         torchvision.transforms.ToTensor(),
        #     ]
        # )

    def __call__(self, x):
        x = x.to(self.device)
        assert x.shape[0] == 4

        if torch.max(x) > 1.1:
            x = x/255.

        if self.random_cropping:
            sample1, sample2 = self.first_pad_then_random_crop(x)
        else:
            sample1 = x
            sample2 = x.clone()

        sample1, sample2 = self.adjust_brightness(sample1, sample2)
        sample1 = self.random_convolution(sample1)
        sample2 = self.random_convolution(sample2)

        mean1 = torch.mean(sample1)
        mean2 = torch.mean(sample2)
        if mean1 < 0.2:
            sample1 += 0.5
        if mean2 < 0.2:
            sample2 += 0.5
        if mean1 > 0.8:
            sample1 -= 0.5
        if mean2 > 0.8:
            sample2 -= 0.5

        sample1 = torch.clamp(sample1, 0, 1.)
        sample2 = torch.clamp(sample2, 0, 1.)

### for plotting / debugging
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
        # random_int = np.random.randint(20)
        # im1.save(f'/home/cathrin/MA/datadump/simclr/'+str(random_int) + '_1.png')
        # im2.save(f'/home/cathrin/MA/datadump/simclr/'+str(random_int) + '_2.png')
        return sample1, sample2

    def first_pad_then_random_crop(self, img_stack):
        img_stack_dim4 = img_stack.unsqueeze(0) # now dim 4
        assert len(img_stack_dim4.shape) == 4, "4 dimensions needed for replicationpad2d (implementation constraint)"
        obs_padded = self.padding(img_stack_dim4).squeeze(0) # dim 3 again
        c, h, w = obs_padded.shape
        # generate random int between 0 and (w-160+1)-1  !
        w_shift = torch.randint(w - 160 + 1, size=(2,), device=self.device)
        h_shift = torch.randint(h - 210 + 1, size=(2,), device=self.device)
        img_stack1 = obs_padded[:, h_shift[0]:(h_shift[0]+210), w_shift[0]:(w_shift[0]+160)]
        img_stack2 = obs_padded[:, h_shift[1]:(h_shift[1]+210), w_shift[1]:(w_shift[1]+160)]
        assert img_stack1.shape == img_stack2.shape == (4, 210, 160), f"shape is {img_stack2.shape}, input shape was: {obs_padded.shape} w: {w_shift[1]}, h: {h_shift[1]}"
        return img_stack1, img_stack2

    def random_convolution(self, imgs):
        '''
        random covolution in "network randomization"

        (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor

        adjusted to just pass one channel per forward-pass. finally stack the outputs together
        '''
        assert len(imgs.shape) == 3
        assert imgs.shape[0] == 4

        img_h, img_w = imgs.shape[-2], imgs.shape[-1]
        num_stack_channel = imgs.shape[-3]

        # initialize random covolution

        torch.nn.init.xavier_normal_(self.rand_conv.weight.data)
        for i in range(num_stack_channel):
            temp_imgs = imgs[i, :, :].unsqueeze(0).unsqueeze(0)

            # pass EACH single frame of the observation through the conv-layer
            rand_out = self.rand_conv(temp_imgs)
            if i == 0:
                total_out = rand_out
            else:
                total_out = torch.cat((total_out, rand_out), 0)
        total_out = total_out.reshape(num_stack_channel, img_h, img_w)
        return total_out.squeeze()

    def adjust_brightness(self, x1, x2):
        brightness_fct = torch.cuda.FloatTensor(2,).uniform_(0.92, 1.08)
        return torchvision.transforms.functional.adjust_brightness(x1, brightness_factor=brightness_fct[0]).to(self.device), torchvision.transforms.functional.adjust_brightness(x2, brightness_factor=brightness_fct[1]).to(self.device)
