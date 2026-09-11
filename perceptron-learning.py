import numpy as np
import matplotlib.pyplot as plt


examples = np.array(
    [
        [
            [1, 1, 1, -1],  # T Shifted Left
            [-1, 1, -1, -1],
            [-1, 1, -1, -1],
            [-1, 1, -1, -1],
        ],
        [
            [-1, 1, 1, 1],  # T Shifted Right
            [-1, -1, 1, -1],
            [-1, -1, 1, -1],
            [-1, -1, 1, -1],
        ],
        [
            [-1, -1, 1, -1],  # J Shifted Left
            [-1, -1, 1, -1],
            [1, -1, 1, -1],
            [1, 1, 1, -1],
        ],
        [
            [-1, -1, -1, 1],  # J Shifted Right
            [-1, -1, -1, 1],
            [-1, 1, -1, 1],
            [-1, 1, 1, 1],
        ],
    ]
)

fig = plt.figure(0, (6, 6))
for i in range(len(examples)):
    fig.add_subplot(2, 2, i + 1)
    plt.imshow(examples[i], cmap="magma")


# numbericallt labels, T=1 J=-1
y = np.array([1, 1, -1, -1])
# reshape each examples into a row and add a 17th col for bias
X = np.hstack((examples.reshape(-1, 16), np.ones((len(y), 1))))

X.shape, y.shape
X

# init weights with zeros
w = np.zeros(17)
lr = 1.0  # learning rate

i = 1
# compute each perceptron output using the dot product
yhat = np.dot(X[i], w)
yhat, y[i]

# updating weights...
w = w + lr * X[i]
w[:16].reshape(4, 4)

i += 1
i

yhat = np.dot(X[i], w)
yhat, y[i]
w = w - lr * X[i]
w[:16].reshape(4, 4)
i += 1
i

yhat = np.dot(X[i], w)  # Compute perceptron output
yhat, y[i]  # Machine outputs +, but we want it to output - (Case 2)
w = (
    w - lr * X[i]
)  # Machine output a +, but we wanted -, so subtract learning rate * examples
w[:16].reshape(4, 4)
i = 0  # We've reached the end of our examples (index 3, so start over)
yhat = np.dot(X[i], w)  # Compute perceptron output
yhat, y[i]  # Machine outputs +, and we want a +, so do not update weights.

for i in range(4):
    yhat = np.dot(X[i], w)  # Compute perceptron output
    print(yhat, y[i])


plt.show()
