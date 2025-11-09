import numpy as np
import matplotlib.pyplot as plt

# Function and its gradient
f = lambda x: (x + 3)**2
grad = lambda x: 2 * (x + 3)

# Gradient Descent
x, lr = 2, 0.1
steps = [x]
for _ in range(50):
    x_new = x - lr * grad(x)
    if abs(x_new - x) < 1e-6: break
    steps.append(x_new)
    x = x_new

print(f"Local minima at x = {x:.4f}")
print(f"Minimum value y = {f(x):.4f}")

# Plot
x_vals = np.linspace(-6, 2, 100)
plt.plot(x_vals, f(x_vals), label='y = (x+3)²')
plt.scatter(steps, [f(i) for i in steps], color='red', label='Steps')
plt.title("Gradient Descent to Find Local Minima")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(); plt.show()
