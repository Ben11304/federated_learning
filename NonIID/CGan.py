import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable


class generator(nn.Module):
    def __init__(self, args):
        super(generator, self).__init__()
        self.input_dim = args.noise_size
        self.output_dim = args.n_features
        self.class_num = args.n_classes
        self.label_emb = nn.Embedding(self.class_num,self.class_num)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
    
        self.model = nn.Sequential(
            *block(self.input_dim + self.class_num, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024,self.output_dim)
        )

    def forward(self, noise ,label):
        x = torch.cat((self.label_emb(label).squeeze(),noise), 1)
        x = self.model(x)
        return x

class discriminator(nn.Module):
    def __init__(self,args):
        super(discriminator, self).__init__()
        self.input_dim = args.n_features
        self.output_dim = args.n_features
        self.class_num = args.n_classes
        self.label_emb = nn.Embedding(self.class_num,self.class_num)

        self.model = nn.Sequential(
            nn.Linear((self.class_num + self.input_dim), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input ,label):
        # Concatenate label embedding and image to produce input
        x = torch.cat((self.label_emb(label).squeeze(), input), 1)
        x = self.model(x)
        return x
    
class CGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.z_dim = args.z_dim
        self.class_num = args.n_class
        # load dataset
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        data=next(iter(self.data_loader))
        #option
        dim=int(data.shape[1])-8

        # networks init
        self.G = generator(args)
        self.D = discriminator(args)
        self.G_optimizer = optim.RMSprop(self.G.parameters(), lr=args.lrG, alpha=0.9)
        self.D_optimizer = optim.RMSprop(self.D.parameters(), lr=args.lrD, alpha=0.9)
        
        self.MSE_loss = torch.nn.MSELoss()
        self.BCE_loss=nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')



    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        #self.y_real_ = Variable(torch.FloatTensor(self.batch_size, 1).fill_(1.0), requires_grad=False)
        #self.y_fake_ = Variable(torch.FloatTensor(self.batch_size, 1).fill_(0.0), requires_grad=False)

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, da in enumerate(self.data_loader):
                x_=da[:,1:].float()
                y_=da[:,:1].int()
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.z_dim))))
                
                # update D network
                self.D.train()
                self.G.eval()
                self.D_optimizer.zero_grad()

                D_real = self.D(x_, y_)
                D_real_loss = self.MSE_loss(D_real, self.y_real_)
                G_ = self.G(z_, y_)
                D_fake = self.D(G_, y_)
                D_fake_loss = self.MSE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                 # update G network
                self.D.eval()
                self.G.train()
                self.G_optimizer.zero_grad()

                z_=torch.rand((self.batch_size, self.z_dim))
                G_ = self.G(z_, y_)
                D_fake = self.D(G_, y_)
                G_loss = self.MSE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

               
                if (iter + 1) == self.data_loader.dataset.__len__() // self.batch_size:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!")
