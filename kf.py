import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

train_df = pd.read_csv("train.csv")
train_labels = train_df['label'].values
train_images = (train_df.iloc[:,1:].values).astype('float32')

class LN5(nn.Module):
    def __init__(self):
        super(LN5, self).__init__()

        self.lc = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
        )

        self.ll = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(400, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),

            nn.Dropout(0.3),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),

            nn.Dropout(0.3),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.lc(x)
        x = x.view(x.size(0), -1)
        x = self.ll(x)
        return x

def train_model(tt_loader, conv_model, optimizer, scheduler, criterion):
    conv_model.train()
    
    for batch_idx, (data, target) in enumerate(tt_loader):
        data = data.unsqueeze(1)
        data, target = data, target
                    
        optimizer.zero_grad()
        output = conv_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    scheduler.step()
    
def evaluate(num_epoch, data_loader, conv_model):
    conv_model.eval()
    loss = 0
    correct = 0
    
    for data, target in data_loader:
        data = data.unsqueeze(1)
        data, target = data, target
        
        output = conv_model(data)     
        loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    loss /= len(data_loader.dataset)
        
    print('{} Average Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)'.format(
        num_epoch, loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

num_model = 10
num_epochs = 100
conv = []

kf = StratifiedKFold(n_splits=num_model, shuffle=True, random_state=123)
criterion = nn.CrossEntropyLoss()

for k, (tr_idx, val_idx) in enumerate(kf.split(train_images, train_labels)):
    
    print('start model {}'.format(k))
    
    conv_model = LN5()
    optimizer = optim.Adam(params=conv_model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    tt_images = train_images[tr_idx]
    tt_labels = train_labels[tr_idx]
    val_images = train_images[val_idx]
    val_labels = train_labels[val_idx]

    tt_images = tt_images.reshape(tt_images.shape[0], 28, 28)
    tt_images_tensor = torch.tensor(tt_images)/255.0
    tt_labels_tensor = torch.tensor(tt_labels)
    tt_tensor = TensorDataset(tt_images_tensor, tt_labels_tensor)
    tt_loader = DataLoader(tt_tensor, batch_size=420, shuffle=True)

    val_images = val_images.reshape(val_images.shape[0], 28, 28)
    val_images_tensor = torch.tensor(val_images)/255.0
    val_labels_tensor = torch.tensor(val_labels)
    val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)
    val_loader = DataLoader(val_tensor, batch_size=420, shuffle=True)

    for n in range(num_epochs):
        train_model(tt_loader, conv_model, optimizer, scheduler, criterion)
        evaluate(n, val_loader, conv_model)

    torch.save(conv_model.state_dict(), 'conv[{}].pkl'.format(k))
    # net.load_state_dict(torch.load('net_params.pkl'))
    
    conv.append(conv_model)

def make_predictions(data_loader):
    conv_model.eval()
    test_preds = torch.LongTensor()
    
    for i, data in enumerate(data_loader):
        data = data.unsqueeze(1)
        output = conv[0](data)
        for idx in range(1, num_model):
            output = output + conv[idx](data)
        preds = output.cpu().data.max(1, keepdim=True)[1]
        test_preds = torch.cat((test_preds, preds), dim=0)
    return test_preds

test_df = pd.read_csv("test.csv")
test_images = (test_df.iloc[:,:].values).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28)
test_images_tensor = torch.tensor(test_images)/255.0
test_loader = DataLoader(test_images_tensor, batch_size=280, shuffle=False)
test_set_preds = make_predictions(test_loader)

submission_df = pd.read_csv("sample_submission.csv")
submission_df['Label'] = test_set_preds.numpy().squeeze()
submission_df.head()
submission_df.to_csv('submission.csv', index=False)
