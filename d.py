import pandas as pd

# CSV dosyasını oku
file_path = r'C:\Users\Helen\Documents\GitHub\deneme\data\CSI300.csv'  # CSV dosyanızın yolu
df = pd.read_csv(file_path)

# Tarih sütununu datetime formatına dönüştür (tarih ve saat dilimi)
df['date'] = pd.to_datetime(df['date'], utc=True)

# Saat dilimi bilgisini kaldır
df['date'] = df['date'].dt.tz_localize(None)

# Yeni formatta tarih oluştur (%Y%m%d)
df['formatted_date'] = df['date'].dt.strftime('%Y%m%d')

# Sonuçları yazdır
print(df[['date', 'formatted_date']])

# İsterseniz sonucu yeni bir CSV dosyasına kaydedebilirsiniz
df.to_csv('yeni_dosya.csv', index=False)
