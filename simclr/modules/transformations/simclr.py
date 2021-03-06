import kornia
import numpy as np
import torchvision
import torch
import torch.nn

from termcolor import colored


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
    A stochastic data augmentation module that transforms any given data batch randomly
    resulting in two correlated views of the same batch.
    """

    def __init__(self, x_shape, sample):

        self.batch_size, self.nr_channels, self.img_height, self.img_width = x_shape
        self.padding = torch.nn.ReplicationPad2d((30, 30, 20, 20))
        self.sample = torch.zeros(
            size=(self.batch_size, 1, self.img_height, self.img_width))

        self.random_crop_nn = torch.nn.Sequential(
            kornia.augmentation.RandomCrop(size=(self.img_height, self.img_width)))


        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Lambda(lambda x: self.padding(x)),
             torchvision.transforms.Lambda(lambda x: self.random_crop(x)),
             # torchvision.transforms.Lambda(lambda x: self.random_crop(x))])
             torchvision.transforms.Lambda(
                 lambda x: self.random_conv_application(x)),
             torchvision.transforms.Lambda(lambda x: self.brightness(x)),
             torchvision.transforms.Lambda(lambda x: self.sanity_checks(x))])
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.rand_conv = torch.nn.Conv2d(
            1, 1, kernel_size=3, bias=False, padding=1).requires_grad_(False).to(self.device)

    def __call__(self, x):
        return self.transforms(x)

    def sanity_checks(self, x):
        diff = torch.max(x) - torch.min(x)
        if diff < 0.15:
            x = x * 2
        mean_x = torch.mean(x)
        if mean_x < 0.2:
            x += 0.5
        if mean_x > 0.8:
            x -= 0.5

        x = torch.clamp(x, 0, 1.)

        return x

    def random_crop(self, x):
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            x[0, :, :, :], output_size=(self.img_height, self.img_width))
        w_shift = torch.randint(x.shape[-1] - self.img_width + 1, size=(1,))
        h_shift = torch.randint(x.shape[-2] - self.img_height + 1, size=(1,))
        x = x[:, :, h_shift[0]: (h_shift[0]+self.img_height),
              w_shift[0]: (w_shift[0]+self.img_width)]
        return x

    def random_conv_application(self, x):
        torch.nn.init.xavier_normal_(self.rand_conv.weight.data)
        for c in range(self.nr_channels):
            x[:, c, :, :] = self.rand_conv(
                x[:, c, :, :].unsqueeze(1)).squeeze()

        return x

    def brightness(self, x):

        brightness_fct = torch.FloatTensor(1,).uniform_(0.92, 1.08)

        x_after_brightness_conv = torchvision.transforms.functional.adjust_brightness(
            x, brightness_factor=brightness_fct.item())

        mean_x_processed = torch.mean(x_after_brightness_conv)
        if mean_x_processed < 0.15:
            print(colored(f"mean is {mean_x_processed}", "green"))
            x_after_brightness_conv = x_after_brightness_conv + 0.3
        elif mean_x_processed > 0.85:
            print(colored(f"mean is {mean_x_processed}", "green"))
            x_after_brightness_conv = x_after_brightness_conv - 0.3
        x_after_brightness_conv = torch.clamp(x_after_brightness_conv, 0, 1.)

        return x_after_brightness_conv

    def dump_batches(self, x, additional_info=""):
        counter = 0
        for b in range(self.batch_size):
            for c in range(self.nr_channels):
                im = x[b, c, :, :]
                im = torchvision.transforms.ToPILImage()(im)
                im.save(
                    f"/home/cathrin/MA/trash/DA/{counter}_{additional_info}.png")
                counter += 1

            if counter > 20:
                break
        print(f"Saved {counter} images")


class TransformsSimCLRAtari:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, width, height, random_cropping):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else 'cpu')
        self.padding = torch.nn.ReplicationPad2d(
            (30, 30, 20, 20)).to(self.device)
        self.random_cropping = random_cropping
        self.rand_conv = torch.nn.Conv2d(
            1, 1, kernel_size=3, bias=False, padding=1).requires_grad_(False)

    def __call__(self, x):
        # x = x.to(self.device)
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

# for plotting / debugging
        # im_sample1 = sample1.squeeze()
        # im1 = im_sample1[0, :, :]
        # im_sample2 = sample2.squeeze()
        # im2 = im_sample2[0, :, :]
        # for i in range(1, 4):
        #     # 3, frameestack, height, width
        #     # via the last axis --> width
        #     im1 = torch.cat((im1, im_sample1[i, :, :]), axis=-1)
        #     # via the last axis --> width
        #     im2 = torch.cat((im2, im_sample2[i, :, :]), axis=-1)
        # im1 = torchvision.transforms.ToPILImage()(im1)
        # im2 = torchvision.transforms.ToPILImage()(im2)
        # random_int = np.random.randint(20)
        # im1.save(f'/home/cathrin/MA/datadump/simclr/' +
        #          str(random_int) + '_1.png')
        # im2.save(f'/home/cathrin/MA/datadump/simclr/' +
        #          str(random_int) + '_2.png')
        return sample1, sample2

    def first_pad_then_random_crop(self, img_stack):
        img_stack_dim4 = img_stack.unsqueeze(0)  # now dim 4
        assert len(
            img_stack_dim4.shape) == 4, "4 dimensions needed for replicationpad2d (implementation constraint)"
        obs_padded = self.padding(img_stack_dim4).squeeze(0)  # dim 3 again
        c, h, w = obs_padded.shape
        # generate random int between 0 and (w-160+1)-1  !
        # , device=self.device)
        w_shift = torch.randint(w - 160 + 1, size=(2,))
        # , device=self.device)
        h_shift = torch.randint(h - 210 + 1, size=(2,))
        img_stack1 = obs_padded[:, h_shift[0]: (
            h_shift[0]+210), w_shift[0]: (w_shift[0]+160)]
        img_stack2 = obs_padded[:, h_shift[1]: (
            h_shift[1]+210), w_shift[1]: (w_shift[1]+160)]
        assert img_stack1.shape == img_stack2.shape == (
            4, 210, 160), f"shape is {img_stack2.shape}, input shape was: {obs_padded.shape} w: {w_shift[1]}, h: {h_shift[1]}"
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
        brightness_fct = torch.FloatTensor(2,).uniform_(0.92, 1.08)
        return torchvision.transforms.functional.adjust_brightness(x1, brightness_factor=brightness_fct[0]), torchvision.transforms.functional.adjust_brightness(x2, brightness_factor=brightness_fct[1])
