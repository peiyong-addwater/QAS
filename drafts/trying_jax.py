import jax.numpy as jnp
import numpy as np
import jax
from jax import custom_jvp
import optax
import functools
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary, Statevector, partial_trace, state_fidelity, DensityMatrix
from qiskit.providers.aer.backends import StatevectorSimulator, AerSimulator, QasmSimulator
from qiskit import Aer, transpile
from jax import jvp, grad

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])



@functools.partial(jax.vmap, in_axes=(None, 0))
def network(params, x):
  return jnp.dot(params, x)

def compute_loss(params, x, y):
  y_pred = network(params, x)
  loss = jnp.mean(optax.l2_loss(y_pred, y))
  return loss

key = jax.random.PRNGKey(42)
target_params = 0.5

# Generate some data.
xs = jax.random.normal(key, (16, 2))
ys = jnp.sum(xs * target_params, axis=-1)

start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)

# Initialize parameters of the model + optimizer.
params = jnp.array([2., 1.])
opt_state = optimizer.init(params)

#print(jnp.zeros((4,2,3)))
a = jnp.zeros((4,2,3))
#a[1,2,1] = np.average([234234, 454, 423413])
#print(a)
# A simple update loop
for i in range(1):
   # print(i)
    #print(params)
    grads = jax.grad(compute_loss)(params, xs, ys)
    #print(type(grads))
    #print(grads)
    #grads = np.array([0.,0.])
    updates, opt_state = optimizer.update(grads, opt_state)
    #print(updates)
    params = optax.apply_updates(params, updates)
    #print(params)
import time
key = jax.random.PRNGKey(1)
#print(key)
k2 = jax.random.PRNGKey(1)
#print(key==k2)
alpha = np.random.randn(12).reshape(3,4)
#print(alpha)
start = time.time()
sampled = jax.random.categorical(key, alpha, axis=1)
#print(sampled, type(sampled))


prob = jax.nn.softmax(alpha, axis=1)
print(prob)
chosen = jnp.argmax(prob, axis=1)
print(chosen)
for c in chosen:
    print(type(c))
#sample = jax.random.choice(key,a=4, p=prob[1])
#print(sample, type(sample), type(int(sample)))
#sums = jnp.sum(prob, axis=1)
#print(sums)
#init_array = jax.numpy.empty((3,4))
#print(init_array)
#jax.ops.index_update(init_array, (1,2), 45)
end = time.time()

print(end-start)

al = [jnp.array(np.random.randn(12).reshape(3,4)) for _ in range(100)]
stacked_al = jnp.stack(al,axis=0)
#print(stacked_al.shape)
avg_al = jnp.mean(stacked_al, axis=0)
#print(avg_al)