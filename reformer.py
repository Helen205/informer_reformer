import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from reformer_pytorch import Reformer
from google.colab import drive
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred):
    """Geliştirilmiş metrik hesaplama fonksiyonu"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Güvenli MAPE hesaplama
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def train_and_evaluate_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, patience=10):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0

    training_start_time = time.time()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    print("Eğitim başlıyor...")
    for epoch in range(num_epochs):
        # Eğitim aşaması
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Test aşaması
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Learning rate güncelleme
        scheduler.step(avg_test_loss)

        # Early stopping kontrolü
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Test Loss: {avg_test_loss:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    training_time = time.time() - training_start_time
    print(f"\nToplam eğitim süresi: {training_time:.2f} saniye")

    # Çıkarım zamanı ölçümü
    print("\nTest verisi üzerinde çıkarım yapılıyor...")
    inference_start_time = time.time()

    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            predictions.extend(output.cpu().numpy())
            true_values.extend(batch_y.numpy())

    inference_time = time.time() - inference_start_time
    print(f"Toplam çıkarım süresi: {inference_time:.2f} saniye")

    # Metrikleri hesapla
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    metrics = calculate_metrics(true_values, predictions)

    # Görselleştirme
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Eğitim Kaybı', color='blue')
    plt.plot(test_losses, label='Test Kaybı', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Eğitim ve Test Kayıpları')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        'training_time': training_time,
        'inference_time': inference_time,
        'metrics': metrics,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'predictions': predictions,
        'true_values': true_values
    }

class ReformerTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model, depth, heads, bucket_size, n_hashes, output_dim):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)

        self.reformer = Reformer(
            dim=d_model,
            depth=depth,
            heads=heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=True  # return_embeddings parametresini kaldırdık
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Overfitting'i önlemek için dropout ekledik
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size = x.shape[0]

        # Embedding katmanı
        x = self.embedding(x)  # (batch_size, seq_length, d_model)

        # Reformer katmanı
        x = self.reformer(x)  # (batch_size, seq_length, d_model)

        # Layer normalization
        x = self.layer_norm(x)

        # Son sekans çıktısını al
        x = x[:, -1, :]  # (batch_size, d_model)

        # Çıktı katmanı
        x = self.fc_out(x)  # (batch_size, output_dim)

        return x
class FinancialDataset(Dataset):
    def __init__(self, X, y):
        # Veri tiplerini ve boyutları kontrol et
        self.X = torch.FloatTensor(X)  # Shape: (N, seq_length, features)
        self.y = torch.FloatTensor(y)  # Shape: (N, features)

        print(f"Dataset X shape: {self.X.shape}")
        print(f"Dataset y shape: {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_sequences(data, bucket_size=8, train=True):
    """
    Veriyi uygun sekans uzunluğunda hazırlar
    """
    seq_length = bucket_size * 2  # 16

    xs = []
    ys = []

    # Örtüşmeyen sekanslar oluştur
    for i in range(0, len(data) - seq_length - 1, 1 if not train else seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        if len(x) == seq_length:
            xs.append(x)
            ys.append(y)

    X = np.array(xs)
    y = np.array(ys)

    return X, y
def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Veri şekillerini kontrol et ve yazdır
            if epoch == 0:
                print(f"Batch X shape: {batch_X.shape}")
                print(f"Batch y shape: {batch_y.shape}")

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)

            # Çıktı şeklini kontrol et
            if epoch == 0:
                print(f"Model output shape: {output.shape}")

            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
features = [
    'Close',
    'High',
    'Low',
    'Open',
    'Volume',
    'SMA_5',
    'SMA_10',
    'SMA_20',
    'SMA_50',
    'RSI_14',
    'Upper_Band',
    'Middle_Band',
    'Lower_Band',
    'MACD',
]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    bucket_size = 8
    seq_length = bucket_size * 2

    # Model parametreleri
    model_params = {
        'input_dim': len(features),
        'd_model': 256,  # 128'den 256'ya artırıldı
        'depth': 6,      # 4'ten 6'ya artırıldı
        'heads': 8,      # 4'ten 8'e artırıldı
        'bucket_size': bucket_size,
        'n_hashes': 4,
        'output_dim': len(features)
    }
    if drive:
      try:
            drive.mount('/content/drive')
      except Exception as e:
            print(f"Drive bağlantı hatası: {e}")

    dosya_yolu = '/content/drive/MyDrive/ders/ALBRK.IS_veri.csv'

    try:
        # 1. Veriyi oku ve ölçeklendir
        df = pd.read_csv(dosya_yolu)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features].values)
        # Veriyi -1 ile 1 arasında normalize et

        # 2. Önce veriyi train ve test olarak böl
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # 3. Train ve test verilerini ayrı ayrı sekanslar haline getir
        X_train, y_train = prepare_sequences(train_data, bucket_size, train=True)
        X_test, y_test = prepare_sequences(test_data, bucket_size, train=False)

        print(f"Eğitim verisi şekli - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Test verisi şekli - X_test: {X_test.shape}, y_test: {y_test.shape}")

        # 4. Dataset'leri oluştur
        train_dataset = FinancialDataset(X_train, y_train)
        test_dataset = FinancialDataset(X_test, y_test)

        # 5. DataLoader'ları oluştur
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            drop_last=True
        )
        # Model oluştur
        model = ReformerTimeSeries(**model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        criterion = nn.MSELoss()

        # Eğitim
        train_model(model, train_loader, optimizer, criterion, num_epochs=100, device=device)
        results = train_and_evaluate_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=200,
        device=device
    )

        print("\nModel Performans Metrikleri:")
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.4f}")

        print("\nDetaylı Sonuçlar:")
        print(f"Eğitim Süresi: {results['training_time']:.2f} saniye")
        print(f"Çıkarım Süresi: {results['inference_time']:.2f} saniye")

        print(f"{metric}: {value:.4f}")

          # Test ve değerlendirme
        model.eval()
        with torch.no_grad():
              test_dataset = FinancialDataset(X_test, y_test)
              test_loader = DataLoader(
                  test_dataset,
                  batch_size=1,
                  shuffle=False,
                  drop_last=True
              )
              predictions = []

              for batch_X, _ in test_loader:
                  batch_X = batch_X.to(device)
                  output = model(batch_X)
                  predictions.append(output.cpu().numpy())

        predictions = np.array(predictions).reshape(-1, len(features))
        predictions = scaler.inverse_transform(predictions)

          # Görselleştirme
        plt.figure(figsize=(15, 6))
        actual_values = df['Close'].values[train_size * seq_length:train_size * seq_length + len(predictions) * seq_length]
        plt.plot(actual_values, label='Gerçek')
        plt.plot(predictions[:, 0], label='Tahmin')
        plt.legend()
        plt.title('Reformer Model - Kapanış Fiyatı Tahminleri')
        plt.show()


    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()