"""
Optimal Over-relaxation parameter discovery
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)
print(f'Using device: {device}')

# Computing ground truth
def gauss_seidel(Nx, Ny, max_iter=5000, tol=1e-6):
    T = np.zeros((Nx, Ny))
    T[:, 0] = 100.0  # Left boundary
    T[0, :] = 100.0  # Bottom boundary
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    for k in range(max_iter):
        T_old = T.copy()
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                T[i, j] = 0.25 * (T_old[i+1, j] + T[i-1, j] + T_old[i, j+1] + T[i, j-1])
        if np.linalg.norm(T - T_old, ord=np.inf) < tol:
            break
    return T


T_gs = gauss_seidel(Nx, Ny) 

class HeatNet(nn.Module):
    def __init__(self):
        super(HeatNet, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class PINNLoss(nn.Module):
    def __init__(self, device):
        super(PINNLoss, self).__init__()
        self.device = device

    def forward(self, net, collocation_points, left_boundary, right_boundary, top_boundary, bottom_boundary, left_boundary_values, right_boundary_values, top_boundary_values, bottom_boundary_values):
        # Collocation points loss (PDE loss)
        collocation_points.requires_grad = True
        print(collocation_points.shape)
        omega = net(collocation_points)
        

        return loss

def train(net, loss_fn, optimizer, num_epochs, collocation_points, left_boundary, right_boundary, top_boundary, bottom_boundary, left_boundary_values, right_boundary_values, top_boundary_values, bottom_boundary_values):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_fn(net, collocation_points, left_boundary, right_boundary, top_boundary, bottom_boundary, left_boundary_values, right_boundary_values, top_boundary_values, bottom_boundary_values)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

Nx, Ny = 50, 50
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)
X, Y = np.meshgrid(x, y)
X_flat = X.flatten()[:, None]
Y_flat = Y.flatten()[:, None]

collocation_points = torch.tensor(np.hstack((X_flat, Y_flat)), dtype=torch.float32, device=device)

left_boundary = torch.tensor(np.hstack((np.zeros_like(y)[:, None], y[:, None])), dtype=torch.float32, device=device)
right_boundary = torch.tensor(np.hstack((np.ones_like(y)[:, None], y[:, None])), dtype=torch.float32, device=device)
top_boundary = torch.tensor(np.hstack((x[:, None], np.ones_like(x)[:, None])), dtype=torch.float32, device=device)
bottom_boundary = torch.tensor(np.hstack((x[:, None], np.zeros_like(x)[:, None])), dtype=torch.float32, device=device)

n_int = len(collocation_points)
n_bc = len(left_boundary)*4

print(n_int,n_bc)
left_boundary_values_np = np.full((Ny, 1), 100.0, dtype=np.float32)
right_boundary_values_np = np.zeros((Ny, 1), dtype=np.float32)
top_boundary_values_np = np.zeros((Nx, 1), dtype=np.float32)
bottom_boundary_values_np = np.full((Nx, 1), 100.0, dtype=np.float32)
left_boundary_values = torch.tensor(left_boundary_values_np, dtype=torch.float32, device=device)
print(left_boundary_values.shape)
right_boundary_values = torch.tensor(right_boundary_values_np, dtype=torch.float32, device=device)
top_boundary_values = torch.tensor(top_boundary_values_np, dtype=torch.float32, device=device)
bottom_boundary_values = torch.tensor(bottom_boundary_values_np, dtype=torch.float32, device=device)

net = HeatNet().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.01)

loss_fn = PINNLoss(device)

num_epochs = 1
train(net, loss_fn, optimizer, num_epochs, collocation_points, left_boundary, right_boundary, top_boundary, bottom_boundary, left_boundary_values, right_boundary_values, top_boundary_values, bottom_boundary_values)

T_pred = net(collocation_points).detach().cpu().numpy().reshape(Nx, Ny)
 

plt.figure(figsize=(18, 5))
levels = np.linspace(-10, 120, 20)

plt.subplot(1, 3, 1)
plt.contourf(X, Y, T_pred, cmap='hot', levels=levels)
plt.colorbar()
plt.title('Temperature Distribution (PINN)')
plt.xlabel('x')
plt.ylabel('y')


plt.subplot(1, 3, 2)
plt.contourf(X, Y, T_gs, cmap='hot', levels=levels)
plt.colorbar()
plt.title('Temperature Distribution (Gauss-Seidel)')
plt.xlabel('x')
plt.ylabel('y')

# Error plot
plt.subplot(1, 3, 3) 
plt.contourf(X, Y, T_gs - T_pred, cmap='hot')
plt.colorbar()
plt.title('Temperature Error (Gauss-Seidel - PINN)')
plt.xlabel('x')
plt.ylabel('y')

# plt.scatter(collocation_points[:,0].detach().cpu().numpy(), collocation_points[:,1].detach().cpu().numpy(), color='blue', label='Collocation Points')
# plt.legend()

plt.tight_layout()
plt.savefig("temperature.png")