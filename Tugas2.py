import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from imageio import imread

# Memuat gambar
jalur_gambar = 'Downloads/Pemandangan.jpg'
gambar = imread(jalur_gambar)

# Konversi gambar ke grayscale menggunakan metode luminositas
gambar_grayscale = np.dot(gambar[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

# Hitung histogram untuk gambar grayscale
histogram, tepi_bin = np.histogram(gambar_grayscale, bins=256, range=(0, 255))

# Menghitung jumlah total piksel dalam gambar
total_piksel = np.sum(histogram)
print(f'Jumlah total piksel: {total_piksel}')

# Mencari intensitas yang dominan
dominasi_intensitas = np.argmax(histogram)
print(f'Intensitas yang dominan: {dominasi_intensitas} dengan frekuensi {histogram[dominasi_intensitas]}')

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(3, 1, height_ratios=[4, 3, 0.4])

# Subplot 1: Gambar Grayscale
ax1 = fig.add_subplot(gs[0])
ax1.imshow(gambar_grayscale, cmap='gray')
ax1.set_title('Gambar yang diubah ke Grayscale', fontsize=14, fontweight='bold', pad=10)
ax1.axis('off')

# Subplot 2: Histogram
ax2 = fig.add_subplot(gs[1])
ax2.bar(tepi_bin[:-1], histogram, width=1, color='steelblue', edgecolor='darkblue')
ax2.set_xlim(0, 255)
ax2.set_ylim(0, max(histogram) + 500)
ax2.tick_params(axis='x', labelbottom=False)
ax2.set_ylabel('Frekuensi', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Subplot 3: Bar Gradasi
ax3 = fig.add_subplot(gs[2])
bar_warna = np.linspace(0, 255, 256).reshape(1, -1) / 255.0
ax3.imshow(bar_warna, aspect='auto', cmap='gray', extent=(0, 255, 0, 1))
ax3.set_xlim(0, 255)
ax3.set_yticks([])
genap_ticks = np.arange(0, 256, 50)
ax3.set_xticks(genap_ticks)
ax3.set_xticklabels([f'{int(i)}' for i in genap_ticks], fontsize=12)
ax3.set_xlabel('Intensitas Grayscale', fontsize=14)

plt.subplots_adjust(hspace=0.1) 
plt.show()