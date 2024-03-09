import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd.variable import Variable

# Định nghĩa Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        validity = self.sigmoid(x)
        return validity

# Định nghĩa Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Định nghĩa bộ dữ liệu tùy chỉnh
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)

# Cài đặt các tham số
input_dim = 28
latent_dim = 10
output_dim = 28
num_epochs = 100
batch_size = 64
lr = 0.0002

# Khởi tạo discriminator và generator
discriminator = Discriminator(input_dim)
generator = Generator(latent_dim, output_dim)

# Định nghĩa hàm loss và optimizer
adversarial_loss = nn.BCELoss()
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
optimizer_G = optim.Adam(generator.parameters(), lr=lr)

# Chuẩn bị dữ liệu
data = [...]  # Dữ liệu đầu vào tabular
labels = [...]  # Nhãn
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Huấn luyện mô hình
for epoch in range(num_epochs):
    for i, (real_data, labels) in enumerate(dataloader):
        batch_size = real_data.size(0)

        # Chuẩn bị dữ liệu thật và dữ liệu giả
        real_data = Variable(real_data.float())
        labels = Variable(labels.float())
        valid = Variable(torch.ones(batch_size, 1))
        fake = Variable(torch.zeros(batch_size, 1))

        # Huấn luyện Discriminator
        optimizer_D.zero_grad()

        # Loss cho dữ liệu thật
        real_pred = discriminator(real_data)
        d_loss_real = adversarial_loss(real_pred, valid)

        # Loss cho dữ liệu giả
        z = Variable(torch.randn(batch_size, latent_dim))
        gen_labels = Variable(torch.randn(batch_size, output_dim))
        fake_data = generator(z)
        fake_pred = discriminator(fake_data)
        d_loss_fake = adversarial_loss(fake_pred, fake)

        # Tổng loss cho Discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Huấn luyện Generator
        optimizer_G.zero_grad()

        # Loss cho Generator
        z = Variable(torch.randn(batch_size, latent_dim))
        gen_labels = Variable(torch.randn(batch_size, output_dim))
        fake_data = generator(z)
        fake_pred = discriminator(fake_data)
        g_loss = adversarial_loss(fake_pred, valid)

        g_loss.backward()
        optimizer_G.step()

        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss.itemLưu ý rằng đoạn mã trên chỉ là một ví dụ cơ bản và cần được tùy chỉnh phù hợp với bài toán của bạn. Bạn cần thay thế dữ liệu và nhãn thật sự vào biến `data` và `labels`, cùng với việc chỉnh sửa các tham số khác như `input_dim`, `latent_dim`, `output_dim`, `num_epochs`, `batch_size`, và `lr` để phù hợp với yêu cầu của bài toán.