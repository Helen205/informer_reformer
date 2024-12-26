import pandas as pd
import os

# CSV dosyasının bulunduğu klasörün yolu
folder_path = r"C:\Users\helen\Desktop\python\veri en son"
files = [ 'ISATR.IS_veri.csv', 'ISCTR.IS_veri.csv', 'YKBNK.IS_veri.csv', 'GARAN.IS_veri.csv',
         'VAKBN.IS_veri.csv', 'TSKB.IS_veri.csv', 'HALKB.IS_veri.csv', 'ALBRK.IS_veri.csv',
         'SKBNK.IS_veri.csv', 'ISBTR.IS_veri.csv', 'ICBCT.IS_veri.csv']

# İşlem yapılacak sütunların listesi
columns_to_remove = ['Adj Close', 'Positive_Close', 'SMA_100','SMA_200', 'MACD_Signal', 'MACD_Hist']

# Dosyalar üzerinde işlem yap
for file in files:
    file_path = os.path.join(folder_path, file)

    if os.path.exists(file_path):
        # Dosyayı oku
        df = pd.read_csv(file_path)

        # Tarih sütununu datetime formatına çevir ve saat dilimi bilgisini kaldır
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df['Date'] = df['Date'].dt.tz_localize(None)

        # Yeni formatta tarih sütunu oluştur ve eski sütunları kaldır
        df['date'] = df['Date'].dt.strftime('%Y%m%d')
        df = df.drop(columns=['Date'] + columns_to_remove, errors='ignore')

        # Güncellenmiş dosyayı kaydet
        df.to_csv(file_path, index=False)
        print(f"Updated data for {file}:")
        print(df.head())
    else:
        print(f"File not found: {file_path}")
