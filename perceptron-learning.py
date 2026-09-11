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


# numerical labels, T=1 J=-1
y = np.array([1, 1, -1, -1])
# reshape each example into a row and add a 17th col for bias
X = np.hstack((examples.reshape(-1, 16), np.ones((len(y), 1))))

# init weights with zeros
w = np.zeros(17)
lr = 1.0  # learning rate

# --- live weight visualization setup ---
plt.ion()
fig2, ax2 = plt.subplots(num=1, figsize=(4, 4))
im = ax2.imshow(w[:16].reshape(4, 4), cmap="RdBu_r", vmin=-4, vmax=4)
cbar = fig2.colorbar(im, ax=ax2)
cbar.set_label("weight value")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("weights before training (click or press a key to start)")
plt.waitforbuttonpress()


def show_weights(title):
    im.set_data(w[:16].reshape(4, 4))
    ax2.set_title(f"{title}  (click/key for next)")
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    plt.waitforbuttonpress()


# --- training loop: standard perceptron update rule ---
update_count = 0
for epoch in range(10):  # safety cap; this dataset converges in a couple epochs
    mistakes = 0
    for i in range(len(y)):
        yhat = np.dot(X[i], w)
        predicted = 1 if yhat >= 0 else -1
        if predicted != y[i]:
            w = w + lr * y[i] * X[i]
            update_count += 1
            mistakes += 1
            show_weights(f"epoch {epoch}, example {i}: update #{update_count}")
    if mistakes == 0:
        print(f"Converged after {update_count} weight update(s), {epoch} full epoch(s).")
        break

plt.ioff()

# --- final check: does every prediction's sign match its label? ---
for i in range(len(y)):
    yhat = np.dot(X[i], w)
    status = "OK" if np.sign(yhat) == y[i] else "WRONG"
    print(f"example {i}: yhat={yhat:6.1f}  y={y[i]:2d}  {status}")

plt.show()
