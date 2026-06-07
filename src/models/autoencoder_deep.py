import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class MaintenanceDeepAutoencoder:
    """
    Deep Denoising Autoencoder para Health Scoring de maquinaria industrial.

    Arquitectura: Input → 64 → 32 → 16 (espacio latente) → 32 → 64 → Input
    Entrenado EXCLUSIVAMENTE con datos de operación NORMAL.
    Genera un Health Score continuo (0-100) basado en el error de reconstrucción:
        Health Score = 100 × (1 - normalized_reconstruction_error)

    Un Health Score bajo indica comportamiento anómalo del equipo,
    sugiriendo degradación o falla inminente.

    Inspirado en: NVIDIA Applications of AI for Anomaly Detection.
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
        self.encoder = None
        self.scaler = StandardScaler()
        self.history = None
        self._max_mse = None  # Para normalización del Health Score

    def _build_model(self, input_dim: int) -> tuple:
        """Construye la arquitectura del Autoencoder simétrico."""
        tf.random.set_seed(self.random_state)

        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.GaussianNoise(0.1)(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu', name='latent_space')(x)

        # Decoder
        x = keras.layers.Dense(32, activation='relu')(encoded)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        decoded = keras.layers.Dense(input_dim, activation='linear')(x)

        autoencoder = keras.Model(inputs, decoded, name='maintenance_deep_autoencoder')
        encoder = keras.Model(inputs, encoded, name='deep_encoder')

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return autoencoder, encoder

    def fit(self, X_train_normal: np.ndarray) -> dict:
        """
        Entrena el Autoencoder con datos de operación NORMAL.
        Calcula los parámetros de normalización para el Health Score.
        """
        print("  🧠 Entrenando Deep Denoising Autoencoder (solo operación normal)...")

        X_scaled = self.scaler.fit_transform(X_train_normal)
        self.model, self.encoder = self._build_model(X_scaled.shape[1])

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        self.history = self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )

        # Calcular parámetros de normalización del Health Score
        reconstructed = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        self._max_mse = np.percentile(mse, 99) * 3  # Margen para datos anómalos

        print(f"  ✅ Autoencoder entrenado ({len(self.history.history['loss'])} épocas)")
        print(f"  📏 MSE base (P99): {np.percentile(mse, 99):.6f}")

        return {
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
        }

    def predict_health_score(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula el Health Score para cada lectura de sensor.
        
        Returns:
            Health Score (0-100): 100 = equipo en perfecto estado, 0 = falla.
        """
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

        # Normalizar a rango [0, 100]
        normalized = np.clip(mse / self._max_mse, 0, 1)
        health_scores = (1 - normalized) * 100

        return health_scores

    def predict_anomaly(self, X: np.ndarray, threshold: float = 50.0) -> tuple:
        """
        Predice anomalías basadas en el Health Score.
        
        Args:
            threshold: Health Score debajo del cual se considera anomalía.
        
        Returns:
            (predicciones binarias, health scores)
        """
        health_scores = self.predict_health_score(X)
        predictions = (health_scores < threshold).astype(int)
        return predictions, health_scores

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Retorna los errores de reconstrucción crudos."""
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        return np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
