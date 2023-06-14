"""
The train.py of DualGAN: 
"""""
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import itertools
import os

from paper_model_reproduce.DualGAN.utils import ReplayBuffer, Logger, weights_init_normal, \
    wasserstein_loss, compute_gradient_penalty
from paper_model_reproduce.DualGAN.models import Generator, Discriminator
from paper_model_reproduce.DualGAN.datasets import ImageDataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Hyperparameters
Hyperparameters = {"input_c_A": 1, "output_c_A": 1, "input_c_B": 1, "output_c_B": 1, "lr": 2e-4, "batch_size": 4,
                   "image_size": 280, "num_epochs": 300}
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
input_A = Tensor(Hyperparameters["batch_size"], Hyperparameters["input_c_A"], Hyperparameters["image_size"],
                 Hyperparameters["image_size"])
input_B = Tensor(Hyperparameters["batch_size"], Hyperparameters["input_c_B"], Hyperparameters["image_size"],
                 Hyperparameters["image_size"])
# target_real = torch.ones(batch_size)
# target_fake = torch.zeros(batch_size)
target_real = Variable(Tensor(Hyperparameters["batch_size"]).fill_(-1.0), requires_grad=False)
target_fake = Variable(Tensor(Hyperparameters["batch_size"]).fill_(1.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Loss weight for gradient penalty
lambda_gp = 10

# Store the loss
loss_GAN_A2B_list = []
loss_D_B_list = []


# Training
############################################
def train_model(netG_A2B, netG_B2A, netD_A, netD_B, criterion_cycle, optimizer_G, optimizer_D_A, optimizer_D_B,
                dataloader, num_epochs, save_models, save_losses):
    logger = Logger(Hyperparameters["num_epochs"], len(dataloader))

    """"     Training Loop       """""

    print("Starting Training Loop...")
    print(torch.cuda.memory_stats())
    # For each epoch
    for epoch in range(num_epochs + 1)[:]:

        # For each batch in the dataloader
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch["A"]))
            # plt.imshow(np.flipud((np.squeeze(real_A.cpu())[0])), cmap="gray")
            # print(np.flipud((np.squeeze(real_A.cpu())[0])).max())
            # plt.show()
            real_B = Variable(input_B.copy_(batch["B"]))
            # plt.imshow(np.flipud((np.squeeze(real_B.cpu())[0])), cmap="gray")
            # print(np.flipud((np.squeeze(real_B.cpu())[0])).max())
            # plt.show()

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = wasserstein_loss(pred_fake, target_real) * 0.01
            # Store loss_GAN_A2B
            loss_GAN_A2B_list.append(loss_GAN_A2B)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = wasserstein_loss(pred_fake, target_real) * 0.01

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0  # cycle loss is given more weight (10-times)
            # than the adversarial loss as described in the paper

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G_A2B = loss_GAN_A2B + loss_cycle_BAB
            loss_G_B2A = loss_GAN_B2A + loss_cycle_ABA
            loss_G = loss_G_A2B + loss_G_B2A
            loss_G.backward()

            optimizer_G.step()
            ####################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = wasserstein_loss(target_real, pred_real) * 0.1

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = wasserstein_loss(pred_fake, target_fake) * 0.1

            # Total loss
            gradient_penalty = compute_gradient_penalty(netD_A, real_A, fake_A)
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5 + lambda_gp * gradient_penalty
            # '0.5': a weighting it used so that updates to the model have half (0.5) the usual effect.
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ####### Discriminator B #######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = wasserstein_loss(pred_real, target_real) * 0.1

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B)
            loss_D_fake = wasserstein_loss(target_fake, pred_fake) * 0.1

            # Total loss
            gradient_penalty = compute_gradient_penalty(netD_B, real_B, fake_B)
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5 + lambda_gp * gradient_penalty
            # '0.5': a weighting it used so that updates to the model have half (0.5) the usual effect.
            # Store loss_D_B
            loss_D_B_list.append(loss_D_B)

            loss_D_B.backward()

            optimizer_D_B.step()

            # # Progress report (http://localhost:8097)
            # # Loss plot
            logger.log(losses={"loss_G": loss_G,
                               "loss_G_GAN": (loss_GAN_A2B + loss_GAN_B2A),
                               "loss_G_cycle": (loss_cycle_ABA + loss_cycle_BAB),
                               "loss_D": (loss_D_A + loss_D_B),

                               "loss_GAN_A2B": loss_GAN_A2B, "loss_GAN_B2A": loss_GAN_B2A,
                               "loss_cycle_ABA": loss_cycle_ABA, "loss_cycle_BAB": loss_cycle_BAB,
                               "loss_G_A2B": loss_G_A2B,
                               "loss_G_B2A": loss_G_B2A,
                               "loss_D_A": loss_D_A, "loss_D_B": loss_D_B
                               },
                       images={"real_A": real_A, "fake_B": fake_B, "recovered_A": recovered_A,
                               "real_B": real_B, "fake_A": fake_A, "recovered_B": recovered_B})

        # Save models checkpoints
        if save_models:
            batch_size = Hyperparameters["batch_size"]
            model_save_path = f"./output/Norm1_41GE_and_75Philips/{epoch}"
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(netG_A2B.state_dict(),
                       model_save_path + f"/netG_A2B_BatchSize={batch_size}_epoch={epoch}.pth")
            torch.save(netG_B2A.state_dict(),
                       model_save_path + f"/netG_B2A_BatchSize={batch_size}_epoch={epoch}.pth")
            torch.save(netD_A.state_dict(),
                       model_save_path + f"/netD_A_BatchSize={batch_size}_epoch={epoch}.pth")
            torch.save(netD_B.state_dict(),
                       model_save_path + f"/netD_B_BatchSize={batch_size}_epoch={epoch}.pth")

        # Save losses arrays
        if save_losses:
            save_path = f"./output/loss_array"
            np.save(os.path.join(save_path, "loss_GAN_A2B"), np.array(loss_GAN_A2B_list))
            np.save(os.path.join(save_path, "loss_D_B"), np.array(loss_D_B_list))


# training
def train(pretrained=False, epoch=126):
    # Networks
    netG_A2B = Generator(input_nc=Hyperparameters["input_c_A"], output_nc=Hyperparameters["output_c_A"])
    netG_B2A = Generator(input_nc=Hyperparameters["input_c_B"], output_nc=Hyperparameters["output_c_B"])
    netD_A = Discriminator(input_nc=Hyperparameters["input_c_A"])
    netD_B = Discriminator(input_nc=Hyperparameters["input_c_B"])

    if pretrained:
        # Initialize the weights with pretrained parameters
        batch_size = Hyperparameters["batch_size"]
        model_save_path = f"./output/Norm1_41GE_and_75Philips/{epoch}"
        netG_A2B.load_state_dict(
            torch.load(model_save_path + f"/netG_A2B_BatchSize={batch_size}_epoch={epoch}.pth"))
        netG_B2A.load_state_dict(
            torch.load(model_save_path + f"/netG_B2A_BatchSize={batch_size}_epoch={epoch}.pth"))
        netD_A.load_state_dict(torch.load(model_save_path + f"/netD_A_BatchSize={batch_size}_epoch={epoch}.pth"))
        netD_B.load_state_dict(torch.load(model_save_path + f"/netD_B_BatchSize={batch_size}_epoch={epoch}.pth"))
    else:
        # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
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

    # # Put networks on GPUs
    # netG_A2B = torch.nn.DataParallel(netG_A2B)
    # netG_B2A = torch.nn.DataParallel(netG_B2A)
    # netD_A = torch.nn.DataParallel(netD_A)
    # netD_B = torch.nn.DataParallel(netD_B)
    # netG_A2B.cuda()
    # netG_B2A.cuda()
    # netD_A.cuda()
    # netD_B.cuda()

    # Put networks on GPU
    if torch.cuda.is_available():
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # # Print the networks
    # print(f"netG_A2B:\n{netG_A2B}")
    # print(f"netG_B2A:\n{netG_B2A}")
    # print(f"netD_A:\n{netD_A}")
    # print(f"netD_B:\n{netD_B}")

    # Losses
    criterion_cycle = torch.nn.L1Loss()

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

    train_model(netG_A2B, netG_B2A, netD_A, netD_B, criterion_cycle, optimizer_G, optimizer_D_A, optimizer_D_B,
                dataloader, num_epochs=Hyperparameters["num_epochs"], save_models=False, save_losses=True)


if __name__ == "__main__":
    train(pretrained=False)
