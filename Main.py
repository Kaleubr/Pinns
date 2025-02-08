import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.special import jn, gamma
from scipy.integrate import quad

# Desativar avisos oneDNN (opcional)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Parâmetros do problema
N = 2  # Dimensão do espaço (N = 2 para 2D)
R = 1.0  # Raio da bola
n = 1  # Ordem da função de Bessel
j_n1 = 3.8317  # Primeiro zero da função de Bessel J_n (para n = 1)

# Solução exata
def solucao_exata(r):
    # Constante de normalização
    def integrando(r):
        return (r**(1 - N / 2) * jn(n, j_n1 * r))**2 * r**(N - 1)
    integral_value, _ = quad(integrando, 0, R)
    c = np.sqrt((gamma(N / 2) * j_n1**2) / (2 * np.pi**(N / 2) * R**2 * integral_value))
    return c * r**(1 - N / 2) * jn(n, j_n1 * r)

# Função de ativação sin
def sin_activation(x):
    return tf.sin(x)

# Rede neural para a solução via PINNs
def neural_net(x, layers):
    for i, size in enumerate(layers):
        x = tf.keras.layers.Dense(size, activation=sin_activation, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    return x

# Função de perda para PINNs
def loss_fn(u_net, lambda_var, x_r, x_b):
    # Perda da equação diferencial
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_r)
        u = u_net(x_r)
        u_r = tape.gradient(u, x_r)
        u_rr = tape.gradient(u_r, x_r)
    # Equação diferencial em coordenadas esféricas
    loss_res = tf.reduce_mean(tf.square(u_rr + (N - 1) / x_r * u_r + lambda_var * u))

    # Perda da condição de contorno
    u_b = u_net(x_b)
    loss_bc = tf.reduce_mean(tf.square(u_b))

    # Perda de normalização
    #loss_norm = tf.square(tf.reduce_mean(u**2) - 1)

    # Liberar a fita persistente
    del tape

    return loss_res + loss_bc #+ loss_norm

# Treinamento da PINN
def train(u_net, lambda_var, x_r, x_b, lr=0.0001, epochs=10000):
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_fn(u_net, lambda_var, x_r, x_b)
        grads = tape.gradient(loss, u_net.trainable_variables + [lambda_var])
        optimizer.apply_gradients(zip(grads, u_net.trainable_variables + [lambda_var]))
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Amostragem de pontos
N_r = 200  # Número de pontos no interior
N_b = 200  # Número de pontos na fronteira
x_r = np.linspace(0.01, R, N_r)[:, np.newaxis]  # Pontos no interior (evitar r = 0)
x_b = np.array([[R] for _ in range(N_b)])  # Pontos na fronteira

# Normalizar os dados de entrada
x_r = x_r / R  # Normaliza para o intervalo [0, 1]

# Converter para tensores do TensorFlow
x_r_tf = tf.convert_to_tensor(x_r, dtype=tf.float32)
x_b_tf = tf.convert_to_tensor(x_b, dtype=tf.float32)

# Construção da rede neural
layers = [1, 50, 50, 1]  # Arquitetura da rede
inputs = tf.keras.Input(shape=(1,))
outputs = neural_net(inputs, layers)
u_net = tf.keras.Model(inputs, outputs)

# Inicialização do autovalor
lambda_var = tf.Variable(1.0, dtype=tf.float32)

# Treinamento
train(u_net, lambda_var, x_r_tf, x_b_tf, lr=0.0001, epochs=10000)

# Pontos para plotagem
r_values = np.linspace(0.01, R, 1000)  # Alta resolução para plotagem
r_values_tf = tf.convert_to_tensor(r_values[:, np.newaxis], dtype=tf.float32)

# Solução exata
u_exata = solucao_exata(r_values)

# Solução via PINNs
u_pinn = u_net(r_values_tf).numpy().flatten()

# Calcular métricas de precisão
mae = np.mean(np.abs(u_pinn - u_exata))
mse = np.mean((u_pinn - u_exata)**2)
r2 = 1 - np.sum((u_pinn - u_exata)**2) / np.sum((u_exata - np.mean(u_exata))**2)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")

# Plotagem das soluções
plt.figure(figsize=(8, 6))
plt.plot(r_values, u_exata, label="Solução Exata", linewidth=2)
plt.plot(r_values, u_pinn, '--', label="Solução PINN", linewidth=2)
plt.xlabel("r")
plt.ylabel("u(r)")
plt.title("Comparação entre Solução Exata e PINN")
plt.legend()
plt.grid()
plt.show()

# Calcular erro relativo
erro_relativo = np.abs(u_pinn - u_exata) / np.abs(u_exata)

# Calcular erro relativo médio (MRE)
mre = np.mean(erro_relativo)

# Calcular precisão em porcentagem
precisao = 100 * (1 - mre)

print(f"Erro Relativo Médio (MRE): {mre}")
print(f"Precisão: {precisao:.2f}%")