import torch
import time
import matplotlib.pyplot as plt
import pytest

from flashlora.attention import FlashAttention

# Setup
model = FlashAttention(dim=512, heads=8, dim_head=64, lora_dim_out=64, lora_r=8).cuda()
sequence_lengths = [2**i for i in range(10, 15)]

# Benchmarking
times = []
for sequence_length in sequence_lengths:
    x = torch.randn(16, sequence_length, 512).cuda()  # batch of 16, sequence length varies, embedding dimension 512
    y = torch.randn(16, sequence_length, 512).cuda()  # target

    # Warmup
    for _ in range(10):
        out = model(x)
        loss = (out - y).sum()
        loss.backward()

    # Measure execution time
    start_time = time.time()
    for _ in range(100):
        out = model(x)
        loss = (out - y).sum()
        loss.backward()
    end_time = time.time()
    times.append(end_time - start_time)

# Comparison
for sequence_length, time in zip(sequence_lengths, times):
    print(f'Sequence length {sequence_length}: {time:.2f} seconds')

# Plotting
plt.plot(sequence_lengths, times)
plt.title('Execution Time Comparison')
plt.xlabel('Sequence Length')
plt.ylabel('Time (seconds)')
plt.show()