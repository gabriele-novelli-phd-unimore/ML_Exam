import tensorflow as tf
import numpy as np
import math
from matplotlib import pyplot as plt
import os

# global useful variables

number_of_classes=10 # mnist has 10 classes
input_size=28 # images will be 28 x 28
latent_size=20 # dimension for latent space

# functions

# this function takes a vector and a matrix and returns a vector
# perpendicular to the columns of the matrix
def orthogonalize_vector(v,M):
    
    if len(v) != M.shape[1]:
        raise ValueError( f"Vector length ({len(v)}) must equal the number of columns of M ({M.shape[1]}).")

    Q, R = np.linalg.qr(M)
    v_perp = v - Q @ (Q.T @ v)

    return v_perp

# this function predicts the probability of the i^th output and the most probable output at point x
def predict(x, i,model):
    x_0 = tf.Variable(tf.reshape(x, (1, latent_size)), dtype=tf.float32)

    p_tot = model(x_0).numpy()[0] # all probabilities at x_0
    prob = model(x_0).numpy()[0, i] # p_i at x_0
    pred = model(x_0).numpy()[0].argmax() # p_max at x_0
    return prob, pred, p_tot

# this function extracts the jacobian of the NN at point x
@tf.function
def extract_gradients(x_input, model):
    x = tf.expand_dims(x_input, axis=0)
    with tf.GradientTape() as tape:
        tape.watch(x)       
        y = model(x, training=False)
    grads = tape.jacobian(y, x)
    return tf.squeeze(grads)

# this function normalizes the jacobian
def normalize(J):
  norm=np.linalg.norm(J)
  if norm<1e-37: # check for explosion
    return J
  else:
    return J/norm

# this function gets all data at x_0
def get_data_at_point(x_0,i,model_log,model_soft):
    p,pred,p_tot=predict(x_0,i,model_soft)
    J = extract_gradients(x_0,model_log)
    J_i = J[i, :].numpy()
    norm=np.linalg.norm(J_i)
    return norm,p,pred,J_i,J,p_tot

# function to save all images
def handling_images(images, filename="output.png"):

    num_images = len(images)
    cols = 10
    rows = math.ceil(num_images / cols)
    fig_height = max(10, rows * 2)
    
    plt.figure(figsize=(20, fig_height), constrained_layout=True)

    for j in range(num_images):
        plt.subplot(rows, cols, j + 1)
        plt.imshow(np.squeeze(images[j]), cmap='gray')
        plt.title(f"{j}")
        plt.axis("off")


    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    full_path = os.path.abspath(filename)
    print(f"File saved to: {full_path}")

# this function computes the rank of J_i
def compute_rank(J_i):
  rank=np.linalg.matrix_rank(J_i)
  s = np.linalg.svd(J_i, compute_uv=False)
  cond = s[0] / s[-1]
  return rank,s,cond

# this functions plots all the singular values of the Jacobian and the probabilities
def handling_SV_and_Probabilities(SV,P):
    
    # SV plot
    plt.figure(figsize=(10, 6))

    n_steps = len(SV)
    n_vals = len(SV[0])

    for i, s in enumerate(SV):
        plt.scatter(np.full(len(s), i), s, s=10)

    for j in range(n_vals):
      plt.plot(
          range(n_steps),
          [SV[i][j] for i in range(n_steps)],
          linewidth=1
      )
    plt.yscale('log')
    plt.title("SV-steps")
    plt.xlabel("Steps")
    plt.ylabel("Singular Values (log)")
    plt.xticks(range(0, n_steps, max(1, n_steps // 20)))
    plt.yticks()
    plt.tight_layout()
    plt.show()

    # P plot
    plt.figure(figsize=(10, 6))

    n_steps = len(P)
    n_vals = len(P[0])

    for j in range(n_vals):
        plt.plot(
            range(n_steps),
            [P[i][j] for i in range(n_steps)],
            linewidth=2,
            label=f"p[{j}]"
        )
    plt.yscale('log')
    plt.title("P-steps")
    plt.xlabel("Steps")
    plt.ylabel("Probabilities (log)")
    plt.xticks(range(0, n_steps, max(1, n_steps // 20)))
    plt.ylim(0, 1)
    plt.legend() 
    plt.tight_layout()
    plt.show()

