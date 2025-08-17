import torch
from torch.autograd import Variable

def minimize_dot_product(X, Y, num_iterations=1000, learning_rate=0.01):
    # Convert X and Y to PyTorch variables
    X = Variable(torch.Tensor(X))
    Y = Variable(torch.Tensor(Y))

    # Initialize variables a and d with random values
    a = Variable(torch.Tensor([0.0]), requires_grad=True)
    d = Variable(torch.Tensor([0.0]), requires_grad=True)

    # Perform gradient descent
    optimizer = torch.optim.Adam([a, d], lr=learning_rate)

    for i in range(num_iterations):
        # Calculate the dot product
        dot_product = torch.dot((X - a).pow(d), (Y - a).pow(d))

        # Compute the gradient
        optimizer.zero_grad()
        dot_product.backward()

        # Update the variables using gradient descent
        optimizer.step()

    # Retrieve the optimized values of a and d
    a_optimized = a.item()
    d_optimized = d.item()

    return a_optimized, d_optimized,dot_product

# Example usage
X = [1, 2, 3]
Y = [4, 5, 6]

a_min, d_min,dot_product = minimize_dot_product(X, Y)

print("Optimized values: a =", a_min, ", d =", d_min)

print(dot_product)
