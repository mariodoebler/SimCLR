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

        # self.random_crop_params = kornia.augmentation.RandomCrop(size=(self.img_height, self.img_width), padding=None, return_transform=True)

        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Lambda(lambda x: self.padding(x)),
             torchvision.transforms.Lambda(lambda x: self.random_crop(x)),
             # torchvision.transforms.Lambda(lambda x: self.random_crop(x))])
             torchvision.transforms.Lambda(lambda x: self.random_conv_application(x))])
        # torchvision.transforms.Lambda(lambda x: self.brightness(x))])
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.rand_conv = torch.nn.Conv2d(
            1, 1, kernel_size=3, bias=False, padding=1).requires_grad_(False).to(self.device)

    def __call__(self, x):
        return self.transforms(x)

    def pad_randomcrop_conv_brightness(self, x):
        padded = self.padding(x)
        x = kornia.augmentation.RandomCrop(
            size=(self.img_height, self.img_width))(padded)
        for b in range(self.batch_size):
            # i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            #     self.sample, output_size=(self.img_height, self.img_width))
            # im1 = torchvision.transforms.ToPILImage()(x[b, :2, :, :].squeeze())
            # im2 = torchvision.transforms.ToPILImage()(
            #     x[b, 2:, :, :] .squeeze())
            # x[b, :2, :, :] = torchvision.transforms.ToTensor()(
            #     torchvision.transforms.functional.crop(im1, i, j, h, w)).unsqueeze(0)
            # # pudb.set_trace()
            # x[b, 2:, :, :] = torchvision.transforms.ToTensor()(
            #     torchvision.transforms.functional.crop(im2, i, j, h, w)).unsqueeze(0)
            brightness_fct = torch.FloatTensor(1,).uniform_(0.9, 1.1)
            # brightness_fct = torch.FloatTensor(
            temp_img = x[b, :, :, :]  # ON GPU

            temp_img = temp_img.unsqueeze(1)
            torch.nn.init.xavier_normal_(self.rand_conv.weight.data)
            temp_img = self.rand_conv(temp_img)

            temp_img = torchvision.transforms.functional.adjust_brightness(
                temp_img.squeeze(), brightness_factor=brightness_fct.item())

            mean_temp = torch.mean(temp_img)
            if mean_temp < 0.15:
                temp_img = temp_img + 0.3
            elif mean_temp > 0.85:
                temp_img = temp_img - 0.3
            temp_img = torch.clamp(temp_img, 0, 1.)
            x[b, :, :, :] = temp_img
        # self.dump_batches(x, "finished")

    def random_crop(self, x):
        # kornia.augmentation.RandomCrop().get_params(self.batch_size, (self.img_height, self.img_width))
        # x = torchvision.transforms.functional.tensor_to_image(x)
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            x[0, :, :, :], output_size=(self.img_height, self.img_width))
        # x_0 = torchvision.transforms.ToPILImage()(x[:, :2, :, :])
        # x_1 = torchvision.transforms.ToPILImage()(x[:, 2:, :, :])
        # x[:, :2, :, :] = torchvision.transforms.function.crop(x_0, i, j, h, w)
        # x[:, 2:, :, :] = torchvision.transforms.function.crop(x_1, i, j, h, w)
        w_shift = torch.randint(x.shape[-1] - self.img_width + 1, size=(1,))
        h_shift = torch.randint(x.shape[-2] - self.img_height + 1, size=(1,))
        # img_stack1, img_stack2 = torch.empty(img_stack.shape, device=self.device), torch.empty(img_stack.shape, device=self.device)
        x = x[:, :, h_shift[0]:(h_shift[0]+self.img_height),
              w_shift[0]:(w_shift[0]+self.img_width)]
        # x[:, c, :, :] = x[:, c, h_shift[:, 1]:(h_shift[:, 1]+210), w_shift[:, 1]:(w_shift[:, 1]+160)]
        # for b in range(self.batch_size):
        #     i1, j1, h1, w1 = torchvision.transforms.RandomCrop.get_params(obs_padded[:, 0, :, :], output_size=(self.img_height, self.img_width))
        #     i2, j2, h2, w2 = torchvision.transforms.RandomCrop.get_params(obs_padded[:, 0, :, :], output_size=(self.img_height, self.img_width))
        #     img_stack1[b, :, :, :] = torchvision.transforms.functional.crop(obs_padded[b, :, :, :], i1, j1, h1, w1)
        #     img_stack2[b, :, :, :] = torchvision.transforms.functional(obs_padded[b, :, :, :], i2, j2, h2, w2)
        # assert img_stack1.shape[1:] == img_stack2.shape[1:] == (4, 210, 160), f"shape is {img_stack2.shape}, input shape was: {obs_padded.shape} w: {w_shift[1]}, h: {h_shift[1]}"
        # return self.random_crop_nn(x)
        return x

    def random_conv_application(self, x):
        torch.nn.init.xavier_normal_(self.rand_conv.weight.data)
        for c in range(self.nr_channels):
            x[:, c, :, :] = self.rand_conv(
                x[:, c, :, :].unsqueeze(1)).squeeze()

        return x

    def brightness(self, x):
        # padded = self.padding(x)
        # padded = kornia.augmentation.RandomCrop(
        #     size=(self.nr_channels, self.img_height, self.img_width))(x)

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

        # for b in range(self.batch_size):
        #     # i, j, h, w = torchvision.transforms.RandomCrop.get_params(
        #     #     self.sample, output_size=(self.img_height, self.img_width))
        #     # im1 = torchvision.transforms.ToPILImage()(x[b, :2, :, :].squeeze())
        #     # im2 = torchvision.transforms.ToPILImage()(
        #     #     x[b, 2:, :, :] .squeeze())
        #     # x[b, :2, :, :] = torchvision.transforms.ToTensor()(
        #     #     torchvision.transforms.functional.crop(im1, i, j, h, w)).unsqueeze(0)
        #     # # pudb.set_trace()
        #     # x[b, 2:, :, :] = torchvision.transforms.ToTensor()(
        #     #     torchvision.transforms.functional.crop(im2, i, j, h, w)).unsqueeze(0)
        #     brightness_fct = torch.FloatTensor(1,).uniform_(0.9, 1.1)
        #     # brightness_fct = torch.FloatTensor(
        #     # self.batch_size,).uniform_(0.92, 1.08)
        #     temp_img = x[b, :, :, :]
        #
        #     temp_img = temp_img.unsqueeze(1)
        #     torch.nn.init.xavier_normal_(self.rand_conv.weight.data)
        #     temp_img = self.rand_conv(temp_img)
        #
        #     temp_img = torchvision.transforms.functional.adjust_brightness(
        #         temp_img.squeeze(), brightness_factor=brightness_fct.item())
        #
        #     mean_temp = torch.mean(temp_img)
        #     if mean_temp < 0.15:
        #         temp_img = temp_img + 0.3
        #     elif mean_temp > 0.85:
        #         temp_img = temp_img - 0.3
        #     temp_img = torch.clamp(temp_img, 0, 1.)
        #     x[b, :, :, :] = temp_img
        # self.dump_batches(x, "finished")

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
        self.rand_conv = torch.nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(
            self.device).requires_grad_(False)
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
        #     im1 = torch.cat((im1, im_sample1[i, :, :]), axis=-1)  # via the last axis --> width
        #     im2 = torch.cat((im2, im_sample2[i, :, :]), axis=-1)  # via the last axis --> width
        # im1 = torchvision.transforms.ToPILImage()(im1)
        # im2 = torchvision.transforms.ToPILImage()(im2)
        # random_int = np.random.randint(20)
        # im1.save(f'/home/cathrin/MA/datadump/simclr/'+str(random_int) + '_1.png')
        # im2.save(f'/home/cathrin/MA/datadump/simclr/'+str(random_int) + '_2.png')
        return sample1, sample2

    def first_pad_then_random_crop(self, img_stack):
        img_stack_dim4 = img_stack.unsqueeze(0)  # now dim 4
        assert len(
            img_stack_dim4.shape) == 4, "4 dimensions needed for replicationpad2d (implementation constraint)"
        obs_padded = self.padding(img_stack_dim4).squeeze(0)  # dim 3 again
        c, h, w = obs_padded.shape
        # generate random int between 0 and (w-160+1)-1  !
        w_shift = torch.randint(w - 160 + 1, size=(2,), device=self.device)
        h_shift = torch.randint(h - 210 + 1, size=(2,), device=self.device)
        img_stack1 = obs_padded[:, h_shift[0]:(
            h_shift[0]+210), w_shift[0]:(w_shift[0]+160)]
        img_stack2 = obs_padded[:, h_shift[1]:(
            h_shift[1]+210), w_shift[1]:(w_shift[1]+160)]
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
        if "cuda" in self.device.type:
            brightness_fct = torch.cuda.FloatTensor(2,).uniform_(0.92, 1.08)
        else:
            brightness_fct = torch.FloatTensor(2,).uniform_(0.92, 1.08)
        return torchvision.transforms.functional.adjust_brightness(x1, brightness_factor=brightness_fct[0]).to(self.device), torchvision.transforms.functional.adjust_brightness(x2, brightness_factor=brightness_fct[1]).to(self.device)
