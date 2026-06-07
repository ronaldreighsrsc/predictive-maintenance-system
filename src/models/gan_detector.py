import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

class MaintenanceGANDetector:
    """
    Generative Adversarial Network (GAN) para Detección de Anomalías.
    
    Arquitectura:
      - Generador: Mapea ruido aleatorio (espacio latente) a la distribución de sensores.
      - Discriminador: Clasifica si una muestra es real (normal) o falsa (generada).
    
    El Discriminador se vuelve experto en la distribución de operación "NORMAL".
    Durante la inferencia, si llega una anomalía, el Discriminador arrojará
    una probabilidad baja (Fake), lo cual usamos como señal de anomalía.
    
    Se incluyen bloques try/except para manejar la inestabilidad típica
    de las GANs tabulares (Mode Collapse o gradientes NaN).
    """

    def __init__(self, latent_dim: int = 16, epochs: int = 100,
                 batch_size: int = 128, learning_rate: float = 1e-4,
                 random_state: int = 42):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.generator = None
        self.discriminator = None
        self.scaler = StandardScaler()
        
        self.g_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5)
        self.bce = keras.losses.BinaryCrossentropy(from_logits=False)
        
        self._base_score = None
        self._max_deviation = None

    def _build_generator(self, output_dim: int) -> keras.Model:
        """Construye el Generador (Ruido -> Datos Fake)."""
        inputs = keras.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(32, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Dense(output_dim, activation='linear')(x)
        return keras.Model(inputs, outputs, name='gan_generator')

    def _build_discriminator(self, input_dim: int) -> keras.Model:
        """Construye el Discriminador (Datos -> Real(1)/Fake(0))."""
        inputs = keras.Input(shape=(input_dim,))
        # Se inyecta un poco de ruido para robustez y evitar overfitting rápido
        x = keras.layers.GaussianNoise(0.05)(inputs)
        x = keras.layers.Dense(64)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(32)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        return keras.Model(inputs, outputs, name='gan_discriminator')

    def fit(self, X_train_normal: np.ndarray) -> dict:
        """
        Entrena la GAN con datos de operación NORMAL.
        """
        print("  🧠 Entrenando GAN Anomaly Detector (solo operación normal)...")
        tf.random.set_seed(self.random_state)
        
        X_scaled = self.scaler.fit_transform(X_train_normal)
        input_dim = X_scaled.shape[1]
        
        self.generator = self._build_generator(input_dim)
        self.discriminator = self._build_discriminator(input_dim)
        
        dataset = tf.data.Dataset.from_tensor_slices(X_scaled).shuffle(buffer_size=1024).batch(self.batch_size)
        
        final_d_loss = 0.0
        final_g_loss = 0.0
        
        # Bloque try/except para manejar Mode Collapse y errores numéricos
        try:
            for epoch in range(self.epochs):
                epoch_d_loss = 0.0
                epoch_g_loss = 0.0
                steps = 0
                
                for real_data in dataset:
                    batch_size_actual = tf.shape(real_data)[0]
                    
                    # 1. Entrenar Discriminador
                    noise = tf.random.normal([batch_size_actual, self.latent_dim])
                    with tf.GradientTape() as d_tape:
                        fake_data = self.generator(noise, training=True)
                        
                        real_output = self.discriminator(real_data, training=True)
                        fake_output = self.discriminator(fake_data, training=True)
                        
                        # Suavizado de etiquetas (label smoothing) para el real
                        d_loss_real = self.bce(tf.ones_like(real_output) * 0.9, real_output)
                        d_loss_fake = self.bce(tf.zeros_like(fake_output), fake_output)
                        d_loss = d_loss_real + d_loss_fake
                        
                    d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
                    self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
                    
                    # 2. Entrenar Generador
                    noise = tf.random.normal([batch_size_actual, self.latent_dim])
                    with tf.GradientTape() as g_tape:
                        fake_data = self.generator(noise, training=True)
                        fake_output = self.discriminator(fake_data, training=True)
                        
                        # Generador quiere que D crea que son reales (1)
                        g_loss = self.bce(tf.ones_like(fake_output), fake_output)
                        
                    g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
                    self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
                    
                    epoch_d_loss += float(d_loss)
                    epoch_g_loss += float(g_loss)
                    steps += 1
                
                avg_d_loss = epoch_d_loss / steps
                avg_g_loss = epoch_g_loss / steps
                final_d_loss = avg_d_loss
                final_g_loss = avg_g_loss
                
                # Comprobación de Mode Collapse o explosión
                if np.isnan(avg_d_loss) or np.isnan(avg_g_loss):
                    raise ValueError("Pérdida (Loss) se volvió NaN. Posible Mode Collapse.")
                    
        except Exception as e:
            print(f"  ⚠️ Advertencia durante entrenamiento GAN: {e}")
            print("  ⚠️ El entrenamiento se detuvo prematuramente para evitar colapso total.")

        print(f"  ✅ GAN entrenada ({self.epochs} épocas). D_loss: {final_d_loss:.4f}, G_loss: {final_g_loss:.4f}")
        
        # Calcular parámetros de normalización del Health Score (basado en desviación)
        anomaly_scores = self.get_reconstruction_errors(X_train_normal) # Raw anomaly score
        self._base_score = np.median(anomaly_scores)
        self._max_deviation = np.percentile(anomaly_scores, 99) - self._base_score
        if self._max_deviation <= 0:
            self._max_deviation = 0.1 # Safety fallback
        print(f"  📏 Median score: {self._base_score:.4f}, Max Deviation (P99): {self._max_deviation:.4f}")

        return {
            'epochs_trained': self.epochs,
            'final_d_loss': final_d_loss,
            'final_g_loss': final_g_loss,
        }

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Para compatibilidad con la API, usamos 1 - D(x) como 'error de reconstrucción'.
        Si D(x) es alto (real/normal), el error es bajo.
        Si D(x) es bajo (fake/anomalía), el error es alto.
        """
        X_scaled = self.scaler.transform(X)
        # Probabilidad de ser "Normal" (0 a 1)
        d_probs = self.discriminator.predict(X_scaled, verbose=0).flatten()
        # Anomaly score (0 a 1)
        anomaly_scores = 1.0 - d_probs
        return anomaly_scores

    def predict_health_score(self, X: np.ndarray) -> np.ndarray:
        """Calcula el Health Score (0-100)."""
        anomaly_scores = self.get_reconstruction_errors(X)
        
        # Medimos la desviación por encima del score normal típico
        deviations = np.maximum(0, anomaly_scores - self._base_score)
        
        # Normalizar asumiendo que 3 veces la desviación máxima de P99 es falla total
        normalized = np.clip(deviations / (self._max_deviation * 3 + 1e-9), 0, 1)
        health_scores = (1 - normalized) * 100
        return health_scores

    def predict_anomaly(self, X: np.ndarray, threshold: float = 50.0) -> tuple:
        """Predice anomalías basadas en el Health Score."""
        health_scores = self.predict_health_score(X)
        predictions = (health_scores < threshold).astype(int)
        return predictions, health_scores
