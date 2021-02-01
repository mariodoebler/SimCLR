import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet, get_resnet_atari
from simclr.modules.transformations import TransformsSimCLR, TransformsSimCLRAtari
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook

import wandb
from atariari.benchmark.episodes import get_episodes
from dataset_atari import AtariDataset


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    # for step, ((x_i, x_j), _) in enumerate(train_loader):
    # len of train_loader: nr_samples / batch_size
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args, wandb=None):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if "NoFrameskip-v4" not in args.env:
        raise NotImplementedError
    # if args.dataset == "STL10":
    #     train_dataset = torchvision.datasets.STL10(
    #         args.dataset_dir,
    #         split="unlabeled",
    #         download=True,
    #         transform=TransformsSimCLR(size=args.image_size),
    #     )
    # elif args.dataset == "CIFAR10":
    #     train_dataset = torchvision.datasets.CIFAR10(
    #         args.dataset_dir,
    #         download=True,
    #         transform=TransformsSimCLR(size=args.image_size),
    #     )
    # else:
    #     raise NotImplementedError
    # def get_episodes(env_name,
    #                  steps,
    #                  seed=42,
    #                  num_processes=1,
    #                  num_frame_stack=1,
    #                  downsample=False,
    #                  color=False,
    #                  entropy_threshold=0.6,
    #                  collect_mode="random_agent",
    #                  train_mode="probe",
    #                  checkpoint_index=-1,
    #                  min_episode_length=64,
    #                  wandb=None,
    #                  use_extended_wrapper=False,
    #                  just_use_one_input_dim=True,
    #                  no_offsets=False):

    if "Breakout" in args.env:
        min_episode_length = 32
    else:
        min_episode_length = 64 # default for mila -> just doesn't work for random Breakout...
    train_episodes = get_episodes(args.env, steps=args.pretraining_steps, seed=args.seed, num_processes=1, num_frame_stack=4, downsample=False, color=False, entropy_threshold=0.6, collect_mode='random_agent', train_mode='train_encoder', min_episode_length=min_episode_length, wandb=wandb, just_use_one_input_dim = False, no_offsets=True, use_extended_wrapper=True, collect_for_curl=True)
    train_dataset = AtariDataset(train_episodes, transform=TransformsSimCLRAtari(width=args.image_width, height=args.image_height))
    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # initialize ResNet
    # encoder = get_resnet(args.resnet, pretrained=False)
    encoder = get_resnet_atari(framestack=4)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 30 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            args.current_epoch += 1

        if wandb:
            wandb.log({'loss': loss_epoch})
            wandb.log({'lr': lr})

    ## end training
    save_model(args, model, optimizer, wandb)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config_atari.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    ## added on my own
    parser.add_argument(
        "--name-logging",
        type=str,
        default=None,
        help="Name which should be used for the logging to wandb."
    )
    parser.add_argument(
        "--project",
        type=str,
        default="trash",
        help="Project Name for WANDB"
    )
    parser.add_argument(
        "--wandb-off",
        action="store_true",
        help="Whether WANDB should NOT log something"
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Which environment, e.g. PongNoFrameskip-v4"
    )

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.wandb_off:
        wandb = None
    else:
        tags = [f"{args.epochs} epochs", args.env, args.resnet, args.device.type]
        wandb = wandb.init(project=args.project, name=args.name_logging, config=args, tags=tags)

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args, wandb=wandb)
