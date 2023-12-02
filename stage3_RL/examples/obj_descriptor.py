import numpy as np
import os 
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.io import read_image
from rrl.multicam import make_encoder



def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


class CombinedEncoder(nn.Module):
    def __init__(self):
        super(CombinedEncoder, self).__init__()
        self.encoder_desc = torch.load('/home/qingyu/forks/H-InDex/archive/resnet.pth')
        self.encoder_desc = nn.DataParallel(self.encoder_desc)

        encoder_ckpt = '/home/qingyu/forks/H-InDex/archive/adapted_frankmocap_hand_ckpts/relocate-large_clamp.pth'
        self.encoder_posenet = make_encoder(encoder_type='hindex', device='cuda', is_eval=True, ckpt_path=encoder_ckpt, test_time_momentum=0.01)
        
        
        self.latent_dim = 2832
        self.cuda()

    def forward(self, x):
        feature1 =  self.encoder_desc(x)
        feature2 = self.encoder_posenet(x)
        return torch.cat((feature1, feature2), dim=1)
    
    def get_features(self,x):
        with torch.no_grad():
            z = self.forward(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/
    
    def get_transform(self):
        return self.encoder_posenet.get_transform()
    


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)  
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32,64, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 3, 2, stride=2)

        self.lk_relu = nn.LeakyReLU(0.1)
        self.resnet50  = models.resnet18(pretrained=True) 
        # for name, child in self.resnet50.named_children():
        #     if name not in ['layer4', 'fc']:
        #         for param in child.parameters():
        #             param.requires_grad = False

        self.resnet50.fc = torch.nn.Linear(self.resnet50.fc.in_features, 4*14*14)

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0),4,14,14)

        
        ## decode ##
        x = self.lk_relu(self.t_conv1(x))
        x = self.lk_relu(self.t_conv2(x))
        x = self.lk_relu(self.t_conv3(x))
        x = torch.sigmoid(self.t_conv4(x))
        # print(x.shape)
        # raise
                
        return x
    
class AdroitImgDataset(Dataset):
    def __init__(self, transform=None):
        self.directory = 'AdroitImgDataset/relocate-large_clamp'
        self.transform = transform
        self.images = glob.glob(self.directory+'/**/*.png')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = read_image(img_path)/255.0

        if self.transform:
            image = self.transform(image)
        label = torch.tensor([0]) 
        return image, label
    
def test_dataset():
    train_data = AdroitImgDataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, num_workers=4)
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
    plt.show()

def test_reconstruction():
    num_workers = 0
    batch_size = 20
    trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224))
    dataset = AdroitImgDataset(transform=trans)
    train_size = int(len(dataset)*0.7)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    model = torch.load('archive/object_descriptor.pth')
    model.eval()
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.cuda()
    output = model(images)
    images = images.detach().cpu().numpy()
    output = output.detach().cpu().numpy()


    bcol = 3 

    fig, axes = plt.subplots(nrows=2, ncols=bcol, sharex=True, sharey=True, figsize=(24,4))

    for idx in np.arange(bcol):
        ax = fig.add_subplot(2, bcol, idx+1, xticks=[], yticks=[])
        ax.imshow(np.transpose(output[idx], (1, 2, 0)))

        ax = fig.add_subplot(2, bcol, idx+bcol+1, xticks=[], yticks=[])
        ax.imshow(np.transpose(images[idx], (1, 2, 0)))

    # plot the first ten input images and then reconstructed images
    # fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
    # for idx in np.arange(20):
    #     ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    #     imshow(images[idx])

    plt.show()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()


    num_workers = 0
    batch_size = 20

    trans = nn.Sequential(transforms.Resize(256),
                                transforms.CenterCrop(224))
    dataset = AdroitImgDataset(transform=trans)
    train_size = int(len(dataset)*0.7)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


    # initialize the NN
    model_save_path = 'archive/object_descriptor.pth'
    if os.path.exists(model_save_path):
        print('load existing model...')
        model = torch.load(model_save_path).to(device)
    else:
        print('creating new model...')
        model = ConvAutoencoder().to(device)
    # print(model)

    criterion = nn.BCELoss()
    # loss = criterion(outputs, images)
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-3)

    # number of epochs to train the model
    n_epochs = 1
    
    for epoch in range(1, n_epochs+1):

        model.train()
        train_loss = 0.0
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
        train_loss = train_loss/len(train_loader)
  
        model.eval()
        with torch.no_grad():  
            test_loss = 0.0
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                test_loss += loss.item()*images.size(0)
            test_loss = test_loss/len(test_loader)
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}\tValidation Loss: {test_loss:.6f}')
        torch.save(model, 'archive/object_descriptor.pth')
        torch.save(model.resnet50,'archive/resnet.pth')

    
def test_encoder():
    encoder_ckpt = 'archive/adapted_frankmocap_hand_ckpts/relocate-large_clamp.pth'
    encoder = make_encoder(encoder_type='hindex', device='cuda', is_eval=True, ckpt_path=encoder_ckpt, test_time_momentum=0.01)

    dataset = AdroitImgDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)
    dataiter = iter(data_loader)
    img = dataiter.next()[0].to('cuda')
    preprocessor = encoder.get_transform()
    img = preprocessor(img)

    features = encoder.get_features(img)
    print(img.shape)
    print(features.shape)

def test_encoder2():
    c_encoder = CombinedEncoder()
    dataset = AdroitImgDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)
    dataiter = iter(data_loader)
    img = dataiter.next()[0].to('cuda')
    preprocessor = c_encoder.get_transform()
    img = preprocessor(img)
    features = c_encoder.get_features(img)
    print(features.shape)


if __name__ == '__main__':
    # test_dataset()
    # train()
    # test_reconstruction()
    test_encoder2()