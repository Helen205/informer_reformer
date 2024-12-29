
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
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from google.colab import files

def calculate_metrics(y_true, y_pred):
    """Geliştirilmiş metrik hesaplama fonksiyonu"""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)


    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def train_and_evaluate_model(model, train_loader, test_loader, optimizer, criterion, num_epochs, device, patience=20):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0

    training_start_time = time.time()

    print("Eğitim başlıyor...")
    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Test
        model.eval()
        test_loss = 0
        all_predictions = []
        all_true_values = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                test_loss += loss.item()

                # Tahminleri ve gerçek değerleri topla
                all_predictions.extend(output.cpu().numpy())
                all_true_values.extend(batch_y.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Test Loss: {avg_test_loss:.4f}')

        # Early stopping
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

    training_time = time.time() - training_start_time

    # Son tahminler
    model.eval()
    final_predictions = []
    final_true_values = []
    inference_start_time = time.time()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            output = model(batch_X)
            # Tüm batch'i topla
            final_predictions.extend(output.cpu().numpy())
            final_true_values.extend(batch_y.numpy())

            # Boyutları kontrol et
            if len(final_predictions) >= len(test_loader.dataset):
                break
    inference_time = time.time() - inference_start_time

    # Boyutları eşitle
    final_predictions = np.array(final_predictions)[:len(test_loader.dataset)]
    final_true_values = np.array(final_true_values)[:len(test_loader.dataset)]

    metrics = calculate_metrics(final_true_values, final_predictions)

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
        'predictions': final_predictions,
        'true_values': final_true_values
    }

def prepare_sequences(data, bucket_size=8, train=True):
    """Geliştirilmiş sekans hazırlama fonksiyonu"""
    seq_length = bucket_size * 2
    xs = []
    ys = []

    # Sadece kapanış fiyatını tahmin edelim
    close_index = features.index('Close')

    for i in range(0, len(data) - seq_length - 1, 1):
        x = data[i:i + seq_length]
        # Sadece bir sonraki kapanış fiyatını tahmin et
        y = data[i + seq_length, close_index]
        if len(x) == seq_length:
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys).reshape(-1, 1)


class ReformerTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model, depth, heads, bucket_size, n_hashes, output_dim=1):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, bucket_size * 2, d_model))

        self.input_dropout = nn.Dropout(0.1)

        self.reformer = Reformer(
            dim=d_model,
            depth=depth,
            heads=heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=True
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_dim)  # output_dim=1
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding
        x = self.input_dropout(x)
        x = self.reformer(x)
        x = self.layer_norm(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x
class FinancialDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)  
        self.y = torch.FloatTensor(y)  

        print(f"Dataset X shape: {self.X.shape}")
        print(f"Dataset y shape: {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_sequences(data, bucket_size=8, train=True):
    """Geliştirilmiş sekans hazırlama fonksiyonu"""
    seq_length = bucket_size * 2
    xs = []
    ys = []

   
    close_index = 0

    for i in range(0, len(data) - seq_length - 1, 1):
        x = data[i:i + seq_length]
        # Sadece kapanış fiyatını al
        y = data[i + seq_length, close_index:close_index+1]  # Tek boyutlu yerine (n,1) şeklinde
        if len(x) == seq_length:
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)
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

    model_params = {
        'input_dim': len(features),
        'd_model': 64,  
        'depth': 3,     
        'heads': 4,     
        'bucket_size': bucket_size,
        'n_hashes': 4,
        'output_dim': 1
    }
    try:
        if drive:
            drive.mount('/content/drive')
    except Exception as e:
        print(f"Drive bağlantı hatası: {e}")

    dosya_yolu = '/content/drive/MyDrive/ders/ALBRK.IS_veri.csv'

    try:
        df = pd.read_csv(dosya_yolu)

        # Kapanış fiyatı için scaler
        close_scaler = MinMaxScaler()
        df['Close_scaled'] = close_scaler.fit_transform(df[['Close']])

        # Diğer özellikler için scaler
        other_features = [f for f in features if f != 'Close']
        scaled_features = {}
        for feature in other_features:
            scaler = MinMaxScaler()
            df[f'{feature}_scaled'] = scaler.fit_transform(df[[feature]])
            scaled_features[feature] = scaler

        # Ölçeklendirilmiş verileri birleştir
        scaled_columns = [f'{f}_scaled' for f in features]
        scaled_data = df[scaled_columns].values

        # Train-test split
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Sekansları hazırla
        X_train, y_train = prepare_sequences(train_data, bucket_size)
        X_test, y_test = prepare_sequences(test_data, bucket_size)

        # Çıktı boyutunu kontrol et
        print(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}")
        print(f"Test shapes - X: {X_test.shape}, y: {y_test.shape}")

        # Dataset'leri oluştur
        train_dataset = FinancialDataset(X_train, y_train)
        test_dataset = FinancialDataset(X_test, y_test)

        # DataLoader'ları oluştur
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
        )

        # Model oluştur
        model = ReformerTimeSeries(**model_params).to(device)

        
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.00005,  
        weight_decay=0.002,  
        betas=(0.9, 0.999)
    )
        criterion = nn.HuberLoss(delta=0.9)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=300,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )

        # Eğitim ve değerlendirme
        results = train_and_evaluate_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=300,
            device=device,
            patience=26
        )

        print("\nModel Performans Metrikleri:")
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.4f}")

        # Tahminleri topla
        model.eval()
        all_predictions = []
        all_true_values = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                output = model(batch_X)
                all_predictions.extend(output.cpu().numpy())
                all_true_values.extend(batch_y.numpy())

        # Numpy array'e çevir
        predictions = np.array(all_predictions)
        true_values = np.array(all_true_values)

        # Ölçeklendirmeyi geri al
        predictions = close_scaler.inverse_transform(predictions)
        true_values = close_scaler.inverse_transform(true_values)

        # Görselleştirme
        plt.figure(figsize=(15, 6))
        plt.plot(true_values, label='Gerçek')
        plt.plot(predictions, label='Tahmin')
        plt.xlabel('Gün')
        plt.ylabel('Fiyat (TL)')
        plt.legend()
        plt.title('Reformer Model - Kapanış Fiyatı Tahminleri')
        plt.show()
        
        model_path = 'model.pth'
        torch.save(model.state_dict(), model_path)
        files.download(model_path)


    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()