import numpy as np
import matplotlib.pyplot as plt

names = ["T shifted left", "T shifted right", "J shifted left", "J shifted right"]

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
    plt.title(names[i])


# numerical labels, T=1 J=-1
y = np.array([1, 1, -1, -1])
# reshape each example into a row and add a 17th col for bias
X = np.hstack((examples.reshape(-1, 16), np.ones((len(y), 1))))

# init weights with zeros
w = np.zeros(17)
lr = 1.0  # learning rate

# --- live weight + current-input visualization setup ---
plt.ion()
fig2, (ax_w, ax_in) = plt.subplots(1, 2, num=1, figsize=(8, 4))

im_w = ax_w.imshow(w[:16].reshape(4, 4), cmap="RdBu_r", vmin=-4, vmax=4)
cbar = fig2.colorbar(im_w, ax=ax_w)
cbar.set_label("weight value")
ax_w.set_xticks([])
ax_w.set_yticks([])
ax_w.set_title("weights")

im_in = ax_in.imshow(np.zeros((4, 4)), cmap="magma", vmin=-1, vmax=1)
ax_in.set_xticks([])
ax_in.set_yticks([])
ax_in.set_title("current input")

fig2.suptitle("click or press a key to start")
plt.waitforbuttonpress()


def show_step(title, i):
    im_w.set_data(w[:16].reshape(4, 4))
    im_in.set_data(X[i, :16].reshape(4, 4))
    ax_in.set_title(f"{names[i]}  (y={y[i]:+d})")
    fig2.suptitle(f"{title}  (click/key for next)")
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
            print(f"epoch {epoch}, {names[i]} (y={y[i]:+d}): misclassified -> update #{update_count}")
            show_step(f"epoch {epoch}, example {i}: update #{update_count}", i)
    if mistakes == 0:
        print(f"Converged after {update_count} weight update(s), {epoch} full epoch(s).")
        break

plt.ioff()

# --- final check: does every prediction's sign match its label? ---
for i in range(len(y)):
    yhat = np.dot(X[i], w)
    status = "OK" if np.sign(yhat) == y[i] else "WRONG"
    print(f"example {i} ({names[i]}): yhat={yhat:6.1f}  y={y[i]:2d}  {status}")

# --- 3D decision-surface landscape of the trained perceptron ---
# The perceptron output is a linear function w . x, so its "landscape" is a
# tilted plane rather than a bowl: one side rises (T territory), the other
# sinks (J territory), with the decision boundary as the flat seam at z=0.
# The 16-D pixel space is reduced to its top 2 principal components (via SVD)
# so the surface can be drawn; the 4 real training examples are plotted at
# their true perceptron output height.
pixels = examples.reshape(-1, 16).astype(float)
mean = pixels.mean(axis=0)
centered = pixels - mean
_, _, Vt = np.linalg.svd(centered, full_matrices=False)
pc1, pc2 = Vt[0], Vt[1]
coords = centered @ np.vstack([pc1, pc2]).T  # (4, 2) PCA coords of the examples

pad = 1.5
p1 = np.linspace(coords[:, 0].min() - pad, coords[:, 0].max() + pad, 40)
p2 = np.linspace(coords[:, 1].min() - pad, coords[:, 1].max() + pad, 40)
P1, P2 = np.meshgrid(p1, p2)

w_pix, bias = w[:16], w[16]
recon = mean + np.einsum("ij,k->ijk", P1, pc1) + np.einsum("ij,k->ijk", P2, pc2)
Z = recon @ w_pix + bias

example_z = X @ w
colors = ["tab:red" if label == 1 else "tab:blue" for label in y]

fig3 = plt.figure(2, (13, 6))

# signed surface: a tilted plane, high = T, low = J
ax3 = fig3.add_subplot(1, 2, 1, projection="3d")
zmax = max(np.abs(Z).max(), 1e-6)
ax3.plot_surface(P1, P2, Z, cmap="RdBu_r", vmin=-zmax, vmax=zmax, alpha=0.85, edgecolor="none")
ax3.contour(P1, P2, Z, levels=[0], colors="k", linewidths=2, offset=0)
ax3.scatter(coords[:, 0], coords[:, 1], example_z, c=colors, s=60, depthshade=False)
for name, (px, py), pz in zip(names, coords, example_z):
    ax3.text(px, py, pz, name)
ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")
ax3.set_zlabel("perceptron output  w . x")
ax3.set_title("signed surface (peaks=T, valleys=J)")

# |w . x| surface: a true bowl-shaped valley along the decision boundary
Z_abs = np.abs(Z)
example_z_abs = np.abs(example_z)
ax4 = fig3.add_subplot(1, 2, 2, projection="3d")
ax4.plot_surface(P1, P2, Z_abs, cmap="viridis", alpha=0.85, edgecolor="none")
ax4.contour(P1, P2, Z_abs, levels=[0], colors="k", linewidths=2, offset=0)
ax4.scatter(coords[:, 0], coords[:, 1], example_z_abs, c=colors, s=60, depthshade=False)
for name, (px, py), pz in zip(names, coords, example_z_abs):
    ax4.text(px, py, pz, name)
ax4.set_xlabel("PC1")
ax4.set_ylabel("PC2")
ax4.set_zlabel("|w . x|  (distance from boundary)")
ax4.set_title("|w . x| surface (valley = decision boundary)")

plt.show()
