"""
The train.py of newline_pipe_1: 
1. model is the modified CycleGAN from train_3_3.
"""""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import itertools
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from modified_01.utils import ReplayBuffer,Logger,weights_init_normal
from modified_01.models import Generator, Discriminator
from modified_01.datasets import ImageDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Hyperparameters
Hyperparameters = {"input_c_A": 1, "output_c_A": 1, "input_c_B": 1, "output_c_B": 1, "lr": 2e-4, "batch_size": 8,
                   "image_size": 280, "num_epochs": 200}
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(Hyperparameters["batch_size"], Hyperparameters["input_c_A"], Hyperparameters["image_size"],
                 Hyperparameters["image_size"])
input_B = Tensor(Hyperparameters["batch_size"], Hyperparameters["input_c_B"], Hyperparameters["image_size"],
                 Hyperparameters["image_size"])
# target_real = torch.ones(batch_size)
# target_fake = torch.zeros(batch_size)
target_real = Variable(Tensor(Hyperparameters["batch_size"]).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(Hyperparameters["batch_size"]).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


# Training
############################################
def train_model(netG_A2B, netG_B2A, netD_A, netD_B, criterion_GAN, criterion_cycle, criterion_identity,
                optimizer_G, optimizer_D_A, optimizer_D_B, dataloader, num_epochs, save_models):

    logger = Logger(Hyperparameters["num_epochs"], len(dataloader))

    """"     Training Loop       """""

    print("Starting Training Loop...")
    print(torch.cuda.memory_stats())
    # For each epoch
    lambd = 1  # decay coefficient for the new Identity Loss
    for epoch in range(num_epochs):

        # For each batch in the dataloader
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch["A"]))
            # plt.imshow(np.flipud((np.squeeze(real_A.cpu())[0])), cmap="gray")
            # print(np.flipud((np.squeeze(real_A.cpu())[0])).min())
            # plt.show()
            real_B = Variable(input_B.copy_(batch["B"]))
            # plt.imshow(np.flipud((np.squeeze(real_B.cpu())[0])), cmap="gray")
            # print(np.flipud((np.squeeze(real_B.cpu())[0])).min())
            # plt.show()

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            # G_B2A(A) should eaqual A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0


            # A new Identity loss
            # G_A2B(A) should equal A if real A is fed
            loss_new_identity_A = criterion_identity(fake_B, real_A) * 5.0 * lambd
            # G_B2A(B) should eaqual B if real B is fed
            loss_new_identity_B = criterion_identity(fake_A, real_B) * 5.0 * lambd

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB \
                     + loss_new_identity_A + loss_new_identity_B
            loss_G.backward()

            optimizer_G.step()
            ####################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ####### Discriminator B #######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()


            # # Progress report (http://localhost:8097)
            # # Loss plot
            logger.log(losses={"loss_G": loss_G,
                               "loss_G_identity": (loss_identity_A + loss_identity_B),
                               "loss_G_new_identity": (loss_new_identity_A + loss_new_identity_B),
                               "loss_G_GAN": (loss_GAN_A2B + loss_GAN_B2A),
                               "loss_G_cycle": (loss_cycle_ABA + loss_cycle_BAB),
                               "loss_D": (loss_D_A + loss_D_B),
                               },
                       images={"real_A": real_A, "same_A": same_A, "fake_B": fake_B, "recovered_A": recovered_A,
                               "real_B": real_B, "same_B": same_B, "fake_A": fake_A, "recovered_B": recovered_B})

        # Save models checkpoints
        if save_models:
            key_word = Hyperparameters["batch_size"]
            torch.save(netG_A2B.state_dict(),
                       f"/home/changxiao/CX/code/CycleGAN_MRI/modified_01/output/netG_A2B_BatchSize={key_word}.pth")
            torch.save(netG_B2A.state_dict(),
                       f"/home/changxiao/CX/code/CycleGAN_MRI/modified_01/output/netG_B2A_BatchSize={key_word}.pth")
            torch.save(netD_A.state_dict(),
                       f"/home/changxiao/CX/code/CycleGAN_MRI/modified_01/output/netD_A_BatchSize={key_word}.pth")
            torch.save(netD_B.state_dict(),
                       f"/home/changxiao/CX/code/CycleGAN_MRI/modified_01/output/netD_B_BatchSize={key_word}.pth")
        print(epoch, lambd)
        if lambd >= 0:
            lambd -= 0.05


# training
def train(pretrained):
    # Networks
    netG_A2B = Generator(input_nc=Hyperparameters["input_c_A"], output_nc=Hyperparameters["output_c_A"])
    netG_B2A = Generator(input_nc=Hyperparameters["input_c_B"], output_nc=Hyperparameters["output_c_B"])
    netD_A = Discriminator(input_nc=Hyperparameters["input_c_A"])
    netD_B = Discriminator(input_nc=Hyperparameters["input_c_B"])

    if pretrained:
        # Initialize the weights with pretrained parameters
        key_word = Hyperparameters["batch_size"]
        netG_A2B.load_state_dict(torch.load(f"output/netG_A2B_BatchSize={key_word}.pth"))
        netG_B2A.load_state_dict(torch.load(f"output/netG_B2A_BatchSize={key_word}.pth"))
        netD_A.load_state_dict(torch.load(f"output/netD_A_BatchSize={key_word}.pth"))
        netD_B.load_state_dict(torch.load(f"output/netD_B_BatchSize={key_word}.pth"))
    else:
        # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)

    # # Put networks on GPUs
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     netG_A2B = torch.nn.DataParallel(netG_A2B)
    #     netG_B2A = torch.nn.DataParallel(netG_B2A)
    #     netD_A = torch.nn.DataParallel(netD_A)
    #     netD_B = torch.nn.DataParallel(netD_B)
    # netG_A2B.cuda()
    # netG_B2A.cuda()
    # netD_A.cuda()
    # netD_B.cuda()

    # Put networks on GPUs
    netG_A2B = torch.nn.DataParallel(netG_A2B)
    netG_B2A = torch.nn.DataParallel(netG_B2A)
    netD_A = torch.nn.DataParallel(netD_A)
    netD_B = torch.nn.DataParallel(netD_B)
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

    # # Put networks on GPU
    # if torch.cuda.is_available():
    #     netG_A2B.cuda()
    #     netG_B2A.cuda()
    #     netD_A.cuda()
    #     netD_B.cuda()

    # # Print the networks
    # print(f"netG_A2B:\n{netG_A2B}")
    # print(f"netG_B2A:\n{netG_B2A}")
    # print(f"netD_A:\n{netD_A}")
    # print(f"netD_B:\n{netD_B}")

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=Hyperparameters["lr"] / 4)
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=Hyperparameters["lr"])
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=Hyperparameters["lr"])

    # Dataset loader
    transform_ = [transforms.ToTensor()]
    dataloader = DataLoader(ImageDataset(transforms_=transform_, unaligned=True),
                            batch_size=Hyperparameters["batch_size"], shuffle=True,
                            num_workers=4, drop_last=True)

    train_model(netG_A2B, netG_B2A, netD_A, netD_B, criterion_GAN, criterion_cycle, criterion_identity,
                optimizer_G, optimizer_D_A, optimizer_D_B, dataloader, num_epochs=Hyperparameters["num_epochs"],
                save_models=True)


if __name__ == "__main__":
    train(pretrained=False)
