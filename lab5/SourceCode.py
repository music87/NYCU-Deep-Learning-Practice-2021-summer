import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torchvision.utils as vutils
from torchvision import transforms
from torchvision.models import vgg19
from torch.utils.data import DataLoader
from dataset import ICLEVRDataset
from evaluator import evaluation_model
import copy
import pdb
seed = 0
random.seed(seed)
torch.manual_seed(seed)

class SRGAN(nn.Module):
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super(SRGAN.FeatureExtractor, self).__init__()
            vgg = vgg19(pretrained=True)
            self.main = nn.Sequential(*list(vgg.features.children())[:18])
        def forward(self, image):
            return self.main(image)

    class ResidualBlock(nn.Module):
        def __init__(self, in_features):
            super(SRGAN.ResidualBlock, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(in_features, in_features, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.BatchNorm2d(in_features, 0.8),
                nn.PReLU(),
                nn.Conv2d(in_features, in_features, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                nn.BatchNorm2d(in_features, 0.8),
            )
        def forward(self, x):
            return x + self.main(x)

    class Generator(nn.Module):
        def __init__(self, image_side_length, feature_map_size, image_channel_size, condition_size, condition_embedding_size, n_residual_blocks=16):
            super(SRGAN.Generator, self).__init__()
            self.condition_embedding = nn.Sequential(
                nn.Linear(condition_size, image_side_length * image_side_length),  # (condition_size, H*W)
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.conv1 = nn.Sequential( nn.Conv2d(image_channel_size+1, feature_map_size, kernel_size=9, stride=1, padding=4),
                                        nn.PReLU())
            self.resBlocks = nn.ModuleList()
            for _ in range(n_residual_blocks):
                self.resBlocks.append(SRGAN.ResidualBlock(feature_map_size))
            self.conv2 = nn.Sequential( nn.Conv2d(feature_map_size, feature_map_size, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(feature_map_size, 0.8))
            self.unsample1 = nn.Sequential( nn.Conv2d(feature_map_size, feature_map_size*4, kernel_size=3, stride=1, padding=1),
                                            nn.BatchNorm2d(feature_map_size*4),
                                            nn.PixelShuffle(upscale_factor=2),
                                            nn.PReLU())
            self.unsample2 = nn.Sequential(nn.Conv2d(feature_map_size, feature_map_size*4, kernel_size=3, stride=1, padding=1),
                                           nn.BatchNorm2d(feature_map_size*4),
                                           nn.PixelShuffle(upscale_factor=2),
                                           nn.PReLU())
            self.conv3 = nn.Sequential( nn.Conv2d(feature_map_size, image_channel_size, kernel_size=9, stride=1, padding=4),
                                        nn.Tanh())

        def forward(self, imageLR_batch, condition_batch):
            cond_embedded_batch = self.condition_embedding(condition_batch)
            batch_size, image_side_length = imageLR_batch.shape[0], int(imageLR_batch.shape[2])
            cond_embedded_batch = cond_embedded_batch.view(batch_size, 1, image_side_length, image_side_length)
            conditional_imageLR_vector_batch = torch.cat((imageLR_batch, cond_embedded_batch), dim=1)
            output1 = self.conv1(conditional_imageLR_vector_batch)
            for resBlock in self.resBlocks:
                output = resBlock(output1)
            output2 = self.conv2(output)
            output = output1 + output2
            output = self.unsample1(output)
            output = self.unsample2(output)
            output = self.conv3(output)
            return output


    class Discriminator(nn.Module):
        def __init__(self, image_channel_size, feature_map_size, loss_GAN_type, condition_size, image_side_length):
            super(SRGAN.Discriminator, self).__init__()
            self.condition_embedding = nn.Sequential(
                nn.Linear(condition_size, image_side_length * image_side_length),  # (condition_size, H*W)
                nn.LeakyReLU(0.2, inplace=True),
            )
            strade_list = [(1,1), (2,2), (1,1), (2,2), (1,1), (2,2), (1,1), (2,2), (1,1), (1,1)]
            padding_list = [(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (0,0), (0,0)]
            channel_list = [image_channel_size+1, feature_map_size, feature_map_size, feature_map_size*2, feature_map_size*2, feature_map_size*4, feature_map_size*4, feature_map_size*8, feature_map_size*8, feature_map_size*16]  # image_channel_size+1 is because of the condition
            self.convs = nn.ModuleList()
            self.convs.append(nn.Sequential(
                nn.Conv2d(channel_list[0], channel_list[1], kernel_size=(3, 3), stride=strade_list[0], padding=padding_list[0], bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ))

            for i in range(1, len(padding_list) - 2):
                self.convs.append(nn.Sequential(
                    nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size=(3, 3), stride=strade_list[i], padding=padding_list[i], bias=False),
                    nn.BatchNorm2d(channel_list[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.last_convs = nn.Sequential(
                nn.Conv2d(channel_list[-2], channel_list[-1], kernel_size=(1, 1), stride=strade_list[-2], padding=padding_list[-2],bias=False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(channel_list[-1], 1, kernel_size=(1, 1), stride=strade_list[-1], padding=padding_list[-1],bias=False)
            )
            if loss_GAN_type == 'GAN':
                self.last_layer = nn.Sigmoid()

        def forward(self, image_batch, condition_batch):
            cond_embedded_batch = self.condition_embedding(condition_batch)
            batch_size, image_side_length = cond_embedded_batch.shape[0], int(math.sqrt(cond_embedded_batch.shape[1]))
            cond_embedded_batch = cond_embedded_batch.view(batch_size, 1, image_side_length, image_side_length)
            output = torch.cat((image_batch, cond_embedded_batch), dim=1)
            for idx, conv in enumerate(self.convs):
                output = conv(output)
            output = self.pool(output)
            output = self.last_convs(output)
            output = self.last_layer(output)
            return output

    def __init__(self, noise_size, loss_GAN_type, condition_size, condition_embedding_size, image_side_length):
        super(SRGAN, self).__init__()
        self.feature_map_size_G = 64
        self.feature_map_size_D = 64
        self.image_channel_size = 3  # image has 3 channels involving R, G, and
        imageLR_side_length = 16
        self.generator = self.Generator(imageLR_side_length, self.feature_map_size_G, self.image_channel_size, condition_size, condition_embedding_size)
        self.discriminator = self.Discriminator(self.image_channel_size, self.feature_map_size_D, loss_GAN_type, condition_size, image_side_length)
        self.feature_extractor = self.FeatureExtractor()

class DCGAN(nn.Module):
    #https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    class Generator(nn.Module):
        def __init__(self, noise_size, feature_map_size, image_channel_size, condition_size, condition_embedding_size):
            super(DCGAN.Generator, self).__init__()
            self.condition_embedding = nn.Sequential(
                nn.Linear(condition_size, condition_embedding_size),
                nn.ReLU(True)
            )
            strade_list = [(1,1), (2,2), (2,2), (2,2), (2,2)]
            padding_list = [(0,0), (1,1), (1,1), (1,1), (1,1)]
            channel_list = [noise_size+condition_embedding_size, feature_map_size*8, feature_map_size*4, feature_map_size*2, feature_map_size]
            self.convTrans = nn.ModuleList()
            for i in range(len(padding_list)-1):
                self.convTrans.append(nn.Sequential(
                    nn.ConvTranspose2d(channel_list[i], channel_list[i+1], kernel_size=(4,4), stride=strade_list[i], padding=padding_list[i], bias=False),
                    nn.BatchNorm2d(channel_list[i+1]),
                    nn.ReLU(inplace=True)
                ))
            self.convTrans.append(nn.ConvTranspose2d(channel_list[-1], image_channel_size, kernel_size=(4,4), stride=strade_list[-1], padding=padding_list[-1], bias=False))
            self.tanh = nn.Tanh()
        def forward(self, noise_vector_batch, condition_batch):
            batch_size = condition_batch.shape[0]
            cond_embedded_batch = self.condition_embedding(condition_batch)
            output = torch.cat((noise_vector_batch, cond_embedded_batch), dim=-1).view(batch_size, -1, 1, 1) # concat condition; reshape because nn.ConvTranspose2d requires input (batch_size, num_channel_in, image_height_in, image_width_in)
            for convTran in self.convTrans:
                output = convTran(output)
            output = self.tanh(output)
            return output

    class Discriminator(nn.Module):
        def __init__(self, image_channel_size, feature_map_size, loss_GAN_type, condition_size, image_side_length):
            super(DCGAN.Discriminator, self).__init__()
            self.condition_embedding = nn.Sequential(
                nn.Linear(condition_size, image_side_length*image_side_length), # (condition_size, H*W)
                nn.LeakyReLU(0.2, inplace=True),
            )
            strade_list = [(2, 2), (2, 2), (2, 2), (2, 2), (1, 1)]
            padding_list = [(1, 1), (1, 1), (1, 1), (1, 1), (0, 0)]
            channel_list = [image_channel_size+1, feature_map_size, feature_map_size*2, feature_map_size*4, feature_map_size*8] # image_channel_size+1 is because of the condition
            self.convs = nn.ModuleList()
            # DCGAN suggests discriminator's input layer shouldn't have batchnorm layer
            # WGAN-GP suggests discriminator should use layer normalization instead of batchnorm
            self.convs.append(nn.Sequential(
                nn.Conv2d(channel_list[0], channel_list[1], kernel_size=(4, 4), stride=strade_list[0],
                          padding=padding_list[0], bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            if loss_GAN_type == 'WGANGP':
                for i in range(1, len(padding_list) - 1):
                    self.convs.append(nn.Sequential(
                        nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size=(4, 4), stride=strade_list[i], padding=padding_list[i], bias=False),
                        nn.LayerNorm([channel_list[i + 1], 2**(5-i), 2**(5-i)]),
                        nn.LeakyReLU(0.2, inplace=True)
                    ))
            else: # GAN, WGAN
                for i in range(1, len(padding_list) - 1):
                    self.convs.append(nn.Sequential(
                        nn.Conv2d(channel_list[i], channel_list[i + 1], kernel_size=(4, 4), stride=strade_list[i], padding=padding_list[i], bias=False),
                        nn.BatchNorm2d(channel_list[i + 1]),
                        nn.LeakyReLU(0.2, inplace=True)
                    ))
            self.convs.append(nn.Conv2d(channel_list[-1], 1, kernel_size=(4, 4), stride=strade_list[-1], padding=padding_list[-1], bias=False))
            if loss_GAN_type == 'GAN':
                self.last_layer = nn.Sigmoid()
            elif loss_GAN_type == 'WGAN' or 'WGANGP':
                self.last_layer = nn.Identity()
            else:
                raise NotImplementedError()

        def forward(self, image_batch, condition_batch):
            cond_embedded_batch = self.condition_embedding(condition_batch)
            batch_size, image_side_length = cond_embedded_batch.shape[0], int(math.sqrt(cond_embedded_batch.shape[1]))
            cond_embedded_batch = cond_embedded_batch.view(batch_size, 1, image_side_length, image_side_length)
            output = torch.cat((image_batch, cond_embedded_batch), dim=1)  # concat condition; reshape because nn.Conv2d requires input (batch_size, num_channel_in, image_height_in, image_width_in)
            for idx, conv in enumerate(self.convs):
                output = conv(output)
            output = self.last_layer(output)
            return output

    def __init__(self, noise_size, loss_GAN_type, condition_size, condition_embedding_size, image_side_length):
        super(DCGAN, self).__init__()
        self.feature_map_size_G = 64
        self.feature_map_size_D = 64
        self.image_channel_size = 3  # image has 3 channels involving R, G, and B
        self.generator = self.Generator(noise_size, self.feature_map_size_G, self.image_channel_size, condition_size, condition_embedding_size)
        self.generator.apply(self.init_weights) # initialize generator's parameters
        self.discriminator = self.Discriminator(self.image_channel_size,  self.feature_map_size_D, loss_GAN_type, condition_size, image_side_length)
        self.discriminator.apply(self.init_weights) # initialize discriminator's parameters

    def forward(self):
        pass

    def init_weights(self, module_layer):
        # normal distribution, DCGAN suggest
        classname = module_layer.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module_layer.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(module_layer.weight.data, 1.0, 0.02)
            nn.init.constant_(module_layer.bias.data, 0)

def generate_noise(batch_size, noise_size):
    return torch.randn(batch_size, noise_size)

def train_one_epoch(epoch, train_dataloader, model, optimizerG, optimizerD, criterion, device, args):
    model.train()
    total_lossD = 0
    total_lossG = 0
    total_predict_real = 0 # D(x)
    total_predict_fake = 0 # D(G(z))
    total_predict_chaotic_real = 0 # D(G(z))

    for iteration, pair_batch in enumerate(train_dataloader):
        batch_size = len(pair_batch[0])
        real_image_batch, condition_batch = pair_batch[0].to(device), pair_batch[1].to(device)
        real_LRimage_batch = transforms.Resize((16, 16), Image.BICUBIC)(real_image_batch) # low resolution
        real = torch.ones(batch_size, dtype=torch.float).to(device)  # 1
        real_noise = torch.normal(torch.ones(batch_size), torch.full((batch_size,), 1e-2)).to(device).clamp_(max=1) # 1 with small noise
        fake_noise = torch.normal(torch.zeros(batch_size), torch.full((batch_size,), 1e-2)).to(device).clamp_(min=0) # 0 with small noise

        # discriminator part
        for _ in range(args.discriminator_RepTimesBalance): # to get more precise critic for WGAN
            noise_vector_batch = generate_noise(batch_size, args.noise_size).to(device)  # z
            if args.architectureGanType == 'SRGAN':
                fake_image_batch = model.generator.forward(real_LRimage_batch, condition_batch)
            else:
                fake_image_batch = model.generator.forward(noise_vector_batch, condition_batch)  # G(z) before updating the generator
            predict_real = model.discriminator.forward(real_image_batch, condition_batch).squeeze() # D(x)
            predict_fake = model.discriminator.forward(fake_image_batch.detach(), condition_batch).squeeze() # D(G(z)) before updating the discriminator ; detach because it's irrelevant to generator's parameter in discriminator updating process
            if args.lossGanType == 'GAN' and (args.architectureGanType == 'DCGAN' or args.architectureGanType == 'SRGAN'):
                lossD = criterion(predict_real, real_noise) + criterion(predict_fake, fake_noise) # H(1, D(x)) + H(0, D(G(z)); nn.BCELoss()'s first/second term should have requires_grad == True/False, the order shouldn't be changed
            elif args.lossGanType == 'WGAN':
                lossD = -predict_real.mean() + predict_fake.mean()
            elif args.lossGanType == 'WGANGP':
                eps = torch.rand(batch_size, 1, 1, 1).to(device)
                interpolot_image_batch = (eps * real_image_batch.data + (1-eps)*fake_image_batch.data).requires_grad_(True) #x
                predict_interplot = model.discriminator.forward(interpolot_image_batch, condition_batch) #y
                gradient = torch.autograd.grad(outputs=predict_interplot, inputs=interpolot_image_batch,
                                               grad_outputs=torch.ones(predict_interplot.size()).to(device),
                                               create_graph=True, retain_graph=True)[0] #dydx
                norm_gradient = torch.sqrt(torch.sum(gradient.view(batch_size, -1) ** 2, dim=1))
                gradient_penalty = torch.mean((norm_gradient - 1) ** 2)
                lossD = -predict_real.mean() + predict_fake.mean() + args.gradientPenaltyWeight*gradient_penalty
            else:
                raise NotImplementedError()
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()
            if args.lossGanType == 'WGAN':
                # clip to range [-args.clipRatio, args.clipRatio]
                for param in model.discriminator.parameters():
                    param.data.clamp_(-args.clipRatio, args.clipRatio)

        # generator part
        for _ in range(args.generator_RepTimesBalance): # to balance the power between discriminator and generator for GAN
            noise_vector_batch = generate_noise(batch_size, args.noise_size).to(device)  # z
            if args.architectureGanType == 'SRGAN':
                fake_image_batch = model.generator.forward(real_LRimage_batch, condition_batch)  # G(z) before updating the generator
            else:
                fake_image_batch = model.generator.forward(noise_vector_batch, condition_batch)  # G(z) before updating the generator
            predict_chaotic_real = model.discriminator.forward(fake_image_batch, condition_batch).squeeze() # D(G(z)) after updating the discriminator ; 所謂混亂真實！中文會變成亂碼不知道這句話什麼時候會消失，反正就是混亂
            if args.lossGanType == 'GAN' and args.architectureGanType == 'DCGAN':
                lossG = criterion(predict_chaotic_real, real) # H(1, D(G(z)) ; fake labels are real for generator cost
            elif args.lossGanType == 'GAN' and args.architectureGanType == 'SRGAN':
                # adversarial loss
                loss_adversarial = criterion(predict_chaotic_real, real)
                # content loss
                model.feature_extractor.eval()
                chaotic_real_features = model.feature_extractor(fake_image_batch)
                real_features = model.feature_extractor(real_image_batch)
                loss_content = nn.L1Loss()(chaotic_real_features, real_features.detach())
                lossG = loss_content + 1e-3 * loss_adversarial
            elif args.lossGanType == 'WGAN' or args.lossGanType == 'WGANGP':
                lossG = -predict_chaotic_real.mean()
            else:
                raise NotImplementedError()
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()

        # update some training messages
        total_lossD += lossD.item()
        total_lossG += lossG.item()
        total_predict_real += predict_real.mean().item()
        total_predict_fake += predict_fake.mean().item()
        total_predict_chaotic_real += predict_chaotic_real.mean().item()
        #print(f"[{iteration + 1}/{len(train_dataloader)}] lossD:{lossD:.4f}, lossG:{lossG:.4f}, D(x):{predict_real.mean().item():.4f}, D(G(z)):({predict_fake.mean().item():.4f}, {predict_chaotic_real.mean().item():.4f})") # to check the balance between discriminator and generator like (lossD, lossG), (discriminator's D(G(z)), generator's D(G(z)))
    print(f"[{epoch+1}/{args.epoch_size}] lossD:{total_lossD/len(train_dataloader):.4f}, lossG:{total_lossG/len(train_dataloader):.4f}, D(x):{total_predict_real/len(train_dataloader):.4f}, D(G(z)):({total_predict_fake/len(train_dataloader):.4f}, {total_predict_chaotic_real/len(train_dataloader):.4f})")
    return total_lossD/len(train_dataloader), total_lossG/len(train_dataloader), total_predict_real/len(train_dataloader), total_predict_fake/len(train_dataloader), total_predict_chaotic_real/len(train_dataloader)

def test_one_epoch(fixed_noise_vector_batch, test_dataloader, model, evaulator, device, model_weights_path, args):
    model.eval()
    if model_weights_path != None:  # the model has been trained well
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))  # load the model weights
    # generator part
    total_score = 0
    with torch.no_grad():
        for condition_batch in test_dataloader:
            condition_batch = condition_batch.to(device)
            if args.architectureGanType == 'SRGAN':
                imageLR_noise_vector = torch.randn(condition_batch.shape[0], 3, 16, 16).to(device)
                fake_image_batch = model.generator.forward(imageLR_noise_vector, condition_batch)
            else:
                fake_image_batch = model.generator.forward(fixed_noise_vector_batch, condition_batch)  # G(z)
            # evaulate
            score = evaulator.eval(fake_image_batch.detach(), condition_batch)
            total_score += score
            # denormalize
            invTrans = transforms.Normalize((-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1 / 0.5, 1 / 0.5, 1 / 0.5))  # becasue x = std*z+mean = std(z+mean/std) = (z-(-mean/std))/(1/std)), where mean=0.5 and std=0.5 defined in dataset.py
            fake_image_batch = invTrans(fake_image_batch)
            # draw and save
            plt.axis("off")
            plt.imshow(np.transpose(vutils.make_grid(fake_image_batch, padding=2, normalize=True).cpu(), (1, 2, 0)))
            vutils.save_image(fake_image_batch.detach(), f"{args.ExperimentReport_folder}/current_image_{args.conditionGanType}_{args.architectureGanType}_{args.lossGanType}.png",  nrow=8, normalize=True)
            plt.show()
            plt.close()
        print(f"score:{total_score/len(test_dataloader):.4f}")
    return total_score/len(test_dataloader)

def train(train_dataloader, test_dataloader, evaluator, model, optimizerG, optimizerD, criterion, fixed_eval_noise_vector, device, args):
    lossD_list = []
    lossG_list = []
    predict_real_list = []
    predict_fake_list = []
    predict_chaotic_real_list = []
    score_list = []
    best_model_weights = None
    best_score = 0
    for epoch in range(args.epoch_size):
        lossD, lossG, predict_real, predict_fake, predict_chaotic_real = train_one_epoch(epoch, train_dataloader, model, optimizerG, optimizerD, criterion, device, args)
        score = test_one_epoch(fixed_eval_noise_vector, test_dataloader, model, evaluator, device, None, args)

        # update and store the current best model
        if score > best_score:
            model_weights_path = f"{args.ModelWeights_folder}/{args.conditionGanType}_{args.architectureGanType}_{args.lossGanType}.pt"
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, model_weights_path)
            best_score = score

        # record some training messages
        lossD_list.append(lossD)
        lossG_list.append(lossG)
        predict_real_list.append(predict_real)
        predict_fake_list.append(predict_fake)
        predict_chaotic_real_list.append(predict_chaotic_real)
        score_list.append(score)
        plot_training_process(epoch, lossD_list, lossG_list, predict_real_list, predict_fake_list, predict_chaotic_real_list, score_list, args)
    return model_weights_path

def plot_training_process(epoch, lossD_list, lossG_list, predict_real_list, predict_fake_list, predict_chaotic_real_list, score_list, args):
    with open(f"{args.ExperimentReport_folder}/training_messages_{args.conditionGanType}_{args.architectureGanType}_{args.lossGanType}.txt","w") as text_file:
        print(f"lossD: {lossD_list}", file=text_file)
        print(f"lossG: {lossG_list}", file=text_file)
        print(f"predict_real D(x): {predict_real_list}", file=text_file)
        print(f"predict_fake D(G(z)): {predict_fake_list}", file=text_file)
        print(f"predict_chaotic_real D(G(z)): {predict_chaotic_real_list}", file=text_file)
        print(f"score: {score_list}", file=text_file)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("Training loss/ratio curve")
    plt.xlabel("epoch(s)")
    ax1.set_ylabel("lossD & lossG")
    ax2.set_ylabel("others")
    ax1.plot(range(epoch + 1), lossD_list, label="lossD", color='brown')
    ax1.plot(range(epoch + 1), lossG_list, label="lossG", color='orange')
    ax2.plot(range(epoch + 1), predict_real_list, linestyle=':', label="predict_real D(x)", color='green')  # D(x)
    ax2.plot(range(epoch + 1), predict_fake_list, linestyle=':', label="predict_fake D(G(z))", color='red')  # D(G(z))
    ax2.plot(range(epoch + 1), predict_chaotic_real_list, linestyle=':', label="predict_chaotic_real D(G(z))", color='blue')  # D(G(z))
    ax2.plot(range(epoch + 1), score_list, label="score", color='purple')
    fig.legend()
    plt.savefig(f"{args.ExperimentReport_folder}/training_process_{args.conditionGanType}_{args.architectureGanType}_{args.lossGanType}.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    # ----------Hyper Parameters----------#
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64) # DCGAN/WGAN suggests batch size should be 128/64
    parser.add_argument('--epoch_size', type=int, default=300)
    parser.add_argument('--image_side_length', type=int, default=64)
    parser.add_argument('--noise_size', type=int, default=100)  # DCGAN suggests noise vector size should be 100
    parser.add_argument('--condition_embedding_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--generator_RepTimesBalance', type=int, default=1)
    parser.add_argument('--discriminator_RepTimesBalance', type=int, default=5) # WGAN/WGANGP suggests n_critic should be 5/5
    parser.add_argument('--gradientPenaltyWeight', type=int, default=10)  # WGANGP suggest lambda should be 10
    parser.add_argument('--LR', type=float, default=1e-4) # DCGAN/SRGAN/WGAN/WGANGP suggests Adam/Adam/RMSProp/Adam's learning rate should be 2e-4/2e-4/5e-5/1e-4
    parser.add_argument('--clipRatio', type=float, default=0.01) # WGAN suggests clip ratio should be 0.01
    parser.add_argument('--beta1', type=float, default=0.5) # DCGAN/WGANGP suggests Adam's beta1 should be 0.5/0.5
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--conditionGanType', type=str, default='cGAN') #'cGAN'
    parser.add_argument('--architectureGanType', type=str, default='DCGAN') # 'DCGAN' or 'SRGAN'
    parser.add_argument('--lossGanType', type=str, default='GAN') #'GAN' or 'WGAN' or 'WGANGP'
    parser.add_argument('--ExperimentReport_folder', type=str, default='./ExperimentReport')
    parser.add_argument('--ModelWeights_folder', type=str, default='./ModelWeights')
    parser.add_argument('--mode', type=str, default='tune') #'tune' , 'demo' , or 'resume'
    args = parser.parse_args()
    print(f"{args.ExperimentReport_folder}: {args.conditionGanType}_{args.architectureGanType}_{args.lossGanType}_{args.epoch_size}epochsize_{args.batch_size}batchsize_{args.generator_RepTimesBalance}genRep_{args.discriminator_RepTimesBalance}disRep_{args.LR}LR")
    # ----------Construct Data Loader----------#
    train_dataset = ICLEVRDataset("./dataset", 'train', image_side_length = args.image_side_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataset = ICLEVRDataset("./dataset", 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    #plt.axis("off")
    #plt.imshow(np.transpose(vutils.make_grid(next(iter(train_dataloader))[0][:64], padding=2, normalize=True).cpu(),(1, 2, 0)))
    #plt.show()

    # ----------Run Model----------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = evaluation_model()
    os.makedirs(args.ExperimentReport_folder, exist_ok=True)
    os.makedirs(args.ModelWeights_folder, exist_ok=True)
    if args.architectureGanType == 'DCGAN' and args.lossGanType == 'GAN':
        # DCGAN suggest optimizer should be Adam
        model = DCGAN(args.noise_size, args.lossGanType, train_dataset.condition_size, args.condition_embedding_size, args.image_side_length).to(device)
        optimizerG = optim.Adam(model.generator.parameters(), args.LR, betas=(args.beta1, args.beta2))
        optimizerD = optim.Adam(model.discriminator.parameters(), args.LR, betas=(args.beta1, args.beta2))
        criterion = nn.BCELoss()
    elif args.architectureGanType == 'DCGAN' and args.lossGanType == 'WGAN':
        # WGAN suggest optimizer should be RMSProp or SGD
        model = DCGAN(args.noise_size, args.lossGanType, train_dataset.condition_size, args.condition_embedding_size, args.image_side_length).to(device)
        optimizerG = optim.RMSprop(model.generator.parameters(), args.LR)
        optimizerD = optim.RMSprop(model.discriminator.parameters(), args.LR)
        criterion = None
    elif args.architectureGanType == 'DCGAN' and args.lossGanType == 'WGANGP':
        model = DCGAN(args.noise_size, args.lossGanType, train_dataset.condition_size, args.condition_embedding_size, args.image_side_length).to(device)
        optimizerG = optim.Adam(model.generator.parameters(), args.LR, betas=(args.beta1, args.beta2))
        optimizerD = optim.Adam(model.discriminator.parameters(), args.LR, betas=(args.beta1, args.beta2))
        criterion = None
    elif args.architectureGanType == 'SRGAN' and args.lossGanType == 'GAN':
        model = SRGAN(args.noise_size, args.lossGanType, train_dataset.condition_size, args.condition_embedding_size, args.image_side_length).to(device)
        optimizerG = optim.Adam(model.generator.parameters(), args.LR, betas=(args.beta1, args.beta2))
        optimizerD = optim.Adam(model.discriminator.parameters(), args.LR, betas=(args.beta1, args.beta2))
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError()
    fixed_eval_noise_vector = generate_noise(len(next(iter(test_dataloader))), args.noise_size).to(device)  # z , just for evaluation, not for training

    if args.mode == 'demo' or args.mode == 'resume':
        model_weights_path = f"{args.ModelWeights_folder}/{args.conditionGanType}_{args.architectureGanType}_{args.lossGanType}.pt"
    if args.mode == 'resume':
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))
    if args.mode == 'tune' or args.mode =='resume':
        model_weights_path = train(train_dataloader, test_dataloader, evaluator, model, optimizerG, optimizerD, criterion, fixed_eval_noise_vector, device, args)
    test_one_epoch(fixed_eval_noise_vector, test_dataloader, model, evaluator, device, model_weights_path, args)