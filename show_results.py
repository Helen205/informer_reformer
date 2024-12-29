import os
import argparse
import numpy as np
from sklearn.metrics import r2_score

def calculate_metrics(predictions, truths):
    mae = np.mean(np.abs(predictions - truths))
    mse = np.mean((predictions - truths)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((truths - predictions) / truths)) * 100
    r2 = r2_score(truths.flatten(), predictions.flatten())
    return mae, mse, rmse, mape, r2

def show_metrics():
    print('\nDetaylı Metrik Sonuçları:')
    print('-' * 80)
    
    for result in sorted(os.listdir('results')):
        try:
            predictions = np.load(f"results/{result}/pred.npy")
            truths = np.load(f"results/{result}/true.npy")
            
            mae, mse, rmse, mape, r2 = calculate_metrics(predictions, truths)
            
            print(f"Model: {result}")
            print(f"MAE : {mae:.4f}")
            print(f"MSE : {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAPE: {mape:.4f}%")
            print(f"R² : {r2:.4f}")
            print('-' * 80)
        except Exception as e:
            continue

def show_pred():
    for result in sorted(os.listdir('results')):
        try:
            predictions = np.load(f"results/{result}/pred.npy")
            print(f"\nTahminler ({result}):")
            for pred in predictions[-1, :, 0]:
                print(f"{float(pred):.4f}", end=' | ')
            print()
        except:
            continue

def show_pred_truths():
    for result in sorted(os.listdir('results')):
        try:
            predictions = np.load(f"results/{result}/pred.npy")
            truths = np.load(f"results/{result}/true.npy")
            
            print(f"\nTahmin ve Gerçek Değerler ({result}):")
            for pred, true in zip(predictions[-1, :, 0], truths[-1, :, 0]):
                print(f"Tahmin: {float(pred):.4f}, Gerçek: {float(true):.4f}")
            print()
        except:
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--act', type=str, default='metrics',
                      choices=['metrics', 'pred', 'pred_truths'],
                      help='Görüntülenecek sonuç tipi')
    args = parser.parse_args()
    
    if args.act == 'metrics':
        show_metrics()
    elif args.act == 'pred':
        show_pred()
    elif args.act == 'pred_truths':
        show_pred_truths()
    else:
        print('Geçersiz argüman.')