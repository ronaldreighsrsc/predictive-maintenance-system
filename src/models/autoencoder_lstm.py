import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

class MaintenanceLSTMAutoencoder:
    """
    LSTM Autoencoder para Health Scoring de maquinaria industrial.

    Arquitectura: Input -> GaussianNoise -> Reshape(3D) -> LSTM Encoder -> RepeatVector -> LSTM Decoder -> Dense
    Entrenado EXCLUSIVAMENTE con datos de operación NORMAL.
    Genera un Health Score continuo (0-100) basado en el error de reconstrucción temporal.
    """

    def __init__(self, encoding_dim: int = 16, epochs: int = 100,
                 batch_size: int = 128, learning_rate: float = 1e-3,
                 random_state: int = 42):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self._max_mse = None

    def _build_model(self, input_dim: int) -> keras.Model:
        """Construye la arquitectura LSTM Autoencoder."""
        tf.random.set_seed(self.random_state)

        # Entrada 3D esperada por LSTM: (batch, timesteps, features)
        # Como usamos datos tabulares donde cada fila tiene el contexto móvil (rolling),
        # tratamos cada fila como una secuencia de longitud 1, timesteps=1.
        inputs = keras.Input(shape=(1, input_dim))
        
        # Inyectar ruido Gaussiano para volverlo "Denoising"
        noisy_inputs = keras.layers.GaussianNoise(0.1)(inputs)
        
        # Encoder LSTM
        encoded = keras.layers.LSTM(32, activation='relu', return_sequences=False)(noisy_inputs)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu', name='latent_space')(encoded)

        # Decoder LSTM
        # Repetir el vector latente para el número de timesteps (1)
        decoded = keras.layers.RepeatVector(1)(encoded)
        decoded = keras.layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
        
        # Capa de salida para reconstruir las características originales
        decoded = keras.layers.TimeDistributed(keras.layers.Dense(input_dim, activation='linear'))(decoded)

        autoencoder = keras.Model(inputs, decoded, name='maintenance_lstm_autoencoder')

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return autoencoder

    def fit(self, X_train_normal: np.ndarray) -> dict:
        """
        Entrena el LSTM Autoencoder con datos de operación NORMAL.
        """
        print("  🧠 Entrenando LSTM Autoencoder (solo operación normal)...")

        X_scaled = self.scaler.fit_transform(X_train_normal)
        
        # Reshape a 3D (batch, timesteps=1, features)
        X_3d = np.expand_dims(X_scaled, axis=1)

        self.model = self._build_model(X_scaled.shape[1])

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        self.history = self.model.fit(
            X_3d, X_3d,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )

        # Calcular parámetros de normalización del Health Score
        reconstructed = self.model.predict(X_3d, verbose=0)
        
        # Error de reconstrucción por muestra
        mse = np.mean(np.power(X_3d - reconstructed, 2), axis=(1, 2))
        self._max_mse = np.percentile(mse, 99) * 3

        print(f"  ✅ LSTM Autoencoder entrenado ({len(self.history.history['loss'])} épocas)")
        print(f"  📏 MSE base (P99): {np.percentile(mse, 99):.6f}")

        return {
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
        }

    def predict_health_score(self, X: np.ndarray) -> np.ndarray:
        """Calcula el Health Score usando reconstrucción LSTM."""
        X_scaled = self.scaler.transform(X)
        X_3d = np.expand_dims(X_scaled, axis=1)
        
        reconstructed = self.model.predict(X_3d, verbose=0)
        mse = np.mean(np.power(X_3d - reconstructed, 2), axis=(1, 2))

        normalized = np.clip(mse / self._max_mse, 0, 1)
        health_scores = (1 - normalized) * 100

        return health_scores

    def predict_anomaly(self, X: np.ndarray, threshold: float = 50.0) -> tuple:
        """Predice anomalías basadas en el Health Score LSTM."""
        health_scores = self.predict_health_score(X)
        predictions = (health_scores < threshold).astype(int)
        return predictions, health_scores

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Retorna los errores de reconstrucción crudos."""
        X_scaled = self.scaler.transform(X)
        X_3d = np.expand_dims(X_scaled, axis=1)
        reconstructed = self.model.predict(X_3d, verbose=0)
        return np.mean(np.power(X_3d - reconstructed, 2), axis=(1, 2))
