import unittest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

from flashlora.attention import FlashAttention

class TestFlashAttentionWithLora(unittest.TestCase):
    def setUp(self):
        self.model = FlashAttention(dim=512, heads=8, dim_head=64, lora_dim_out=64, lora_r=8)
        self.model.to('cuda')

    def test_forward(self):
        x = torch.randn(16, 50, 512).to('cuda')  # batch of 16, sequence length 50, embedding dimension 512
        out = self.model(x)
        self.assertEqual(out.shape, (16, 50, 512))

    def test_performance(self):
        # Use MNIST for simplicity
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = F.nll_loss

        epochs = 10
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for data, target in train_loader:
                data, target = data.to('cuda'), target.to('cuda')
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(train_loader))

        plt.plot(losses)
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

if __name__ == '__main__':
    unittest.main()