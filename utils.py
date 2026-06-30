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
def orthogonalize_vector(v, J, tol=1e-30):
    v = np.asarray(v, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    M = J.T  # shape (n, k)
    n, k = M.shape
    Q_cols = []

    for i in range(k):
        q = M[:, i].copy()
        for q_prev in Q_cols:
            q -= (q_prev @ M[:, i]) * q_prev
        norm = np.linalg.norm(q)
        if norm > tol:
            Q_cols.append(q / norm)
            
    v_orth = v.copy()
    for q in Q_cols:
        v_orth -= (q @ v) * q

    norm = np.linalg.norm(v_orth)
    if norm < tol:
        return v

    return (v_orth / norm).astype(np.float32)
    
# this function predicts the probability of the i^th output and the most probable output at point x
def predict(x, i,model):
    x = tf.expand_dims(x, axis=0)
    p_tot = model(x).numpy()[0] # all probabilities at x_0
    prob = model(x).numpy()[0, i] # p_i at x_0
    pred = model(x).numpy()[0].argmax() # p_max at x_0
    return prob, pred, p_tot

# this function extracts the jacobian of the NN at point x
@tf.function
def extract_gradients(x, model):
    x = tf.expand_dims(x, axis=0)
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
def get_data_at_point(x,i,model_log,model_soft):
    p,pred,p_tot=predict(x,i,model_soft)
    J = extract_gradients(x,model_log)
    J_i = J[i, :].numpy()
    norm=np.linalg.norm(J_i)
    return norm,p,pred,J_i,J,p_tot

# function to save all images
def handling_images(images,probs,preds,N, filename="output.png"):

    num_images = len(images)
    cols = 10
    rows = math.ceil(num_images / cols)
    fig_height = max(10, rows * 2)
    
    plt.figure(figsize=(20, fig_height), constrained_layout=True)

    for j in range(num_images):

        n=N[j]
        plt.subplot(rows, cols, j + 1)
        plt.imshow(np.squeeze(images[j]), cmap='gray')
        plt.title(f"Step {n}\n Predicted: {preds[j]}\n With prob : {probs[j]:.3f}",fontsize=8)
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
def handling_SV_and_Probabilities(SV, P):
    # SV plot
    plt.figure(figsize=(10, 6))

    n_steps = len(SV)
    n_vals = len(SV[0])
    max_sv = np.max(SV)
    min_sv = np.min(SV)
    
    for j in range(n_vals):
        sv_path = [SV[i][j] for i in range(n_steps)]
        plt.plot(
            range(n_steps),
            sv_path,
            marker='o',       
            markersize=3,     
            linewidth=1,      
            label=f"SV {j}"   
        )

    plt.yscale('log')
    plt.title("SV-steps")
    plt.xlabel("Steps")
    plt.ylabel("Singular Values (log)")
    plt.xticks(range(0, n_steps, max(1, n_steps // 20)))
    plt.ylim(bottom=min_sv * 0.5, top=max_sv * 2.0)
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
    plt.ylim(top=1.5) 
    
    plt.legend()
    plt.tight_layout()
    plt.show()
