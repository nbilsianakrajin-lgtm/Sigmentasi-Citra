import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import 1_metode_segmentasi as metode

class AplikasiSegmentasi:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Segmentasi Citra - UTS")
        self.root.geometry("900x600")

        self.img_path = None
        self.cv_img = None
        
        # --- Panel Kiri (Tombol & Kontrol) ---
        panel_kiri = tk.Frame(root, width=280, bg="#f0f0f0")
        panel_kiri.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(panel_kiri, text="PILIH METODE:", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        
        self.metode_var = tk.StringVar(value="Thresholding")
        metode_pilihan = ["Thresholding", "Segmentasi Hue", "Deteksi Tepi", "Transformasi Hough"]
        for m in metode_pilihan:
            tk.Radiobutton(panel_kiri, text=m, variable=self.metode_var, value=m, bg="#f0f0f0", font=("Arial", 10)).pack(anchor="w")

        tk.Button(panel_kiri, text="1. Buka Gambar dari Dataset", command=self.buka_gambar, bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(pady=(20, 5), fill=tk.X)
        tk.Button(panel_kiri, text="2. Proses Metode Terpilih", command=self.proses_gambar, bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(pady=5, fill=tk.X)
        tk.Button(panel_kiri, text="3. Simpan Hasil (Grid)", command=self.simpan_grid, bg="#FF9800", fg="white", font=("Arial", 10, "bold")).pack(pady=5, fill=tk.X)
        
        tk.Label(panel_kiri, text="Info:\nGunakan 'Simpan Hasil' untuk\nmenyimpan gabungan 4 metode.", bg="#f0f0f0", font=("Arial", 8, "italic")).pack(pady=20)

        # --- Panel Kanan (Tampilan Gambar) ---
        self.panel_kanan = tk.Frame(root)
        self.panel_kanan.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.lbl_asli = tk.Label(self.panel_kanan, text="Gambar Asli", bg="gray")
        self.lbl_asli.place(relx=0.25, rely=0.5, anchor=tk.CENTER, width=350, height=400)

        self.lbl_hasil = tk.Label(self.panel_kanan, text="Hasil Segmentasi", bg="gray")
        self.lbl_hasil.place(relx=0.75, rely=0.5, anchor=tk.CENTER, width=350, height=400)

    def buka_gambar(self):
        # Default membuka folder dataset jika ada
        initial_dir = "dataset" if os.path.exists("dataset") else "."
        file_path = filedialog.askopenfilename(initialdir=initial_dir, title="Pilih Gambar", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.img_path = file_path
            self.cv_img = cv2.imread(file_path)
            self.tampilkan_gambar(self.cv_img, self.lbl_asli)

    def proses_gambar(self):
        if self.cv_img is None:
            messagebox.showwarning("Peringatan", "Silakan buka gambar terlebih dahulu!")
            return

        m = self.metode_var.get()
        if m == "Thresholding": hasil = metode.proses_thresholding(self.cv_img)
        elif m == "Segmentasi Hue": hasil = metode.proses_hue(self.cv_img)
        elif m == "Deteksi Tepi": hasil = metode.proses_tepi(self.cv_img)
        elif m == "Transformasi Hough": hasil = metode.proses_hough(self.cv_img)

        self.hasil_cv = hasil
        self.tampilkan_gambar(hasil, self.lbl_hasil)

    def tampilkan_gambar(self, img_cv, label):
        if len(img_cv.shape) == 2: img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        else: img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((350, 400))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        label.config(image=img_tk, text="")
        label.image = img_tk

    def simpan_grid(self):
        if self.cv_img is None:
            messagebox.showwarning("Peringatan", "Buka gambar terlebih dahulu!")
            return

        # Menghasilkan gambar grid (Asli + 4 Metode)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Hasil UTS: {os.path.basename(self.img_path)}', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        axes[0].imshow(cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)); axes[0].set_title('Asli'); axes[0].axis('off')
        axes[1].imshow(metode.proses_thresholding(self.cv_img), cmap='gray'); axes[1].set_title('Thresholding'); axes[1].axis('off')
        axes[2].imshow(cv2.cvtColor(metode.proses_hue(self.cv_img), cv2.COLOR_BGR2RGB)); axes[2].set_title('Hue'); axes[2].axis('off')
        axes[3].imshow(metode.proses_tepi(self.cv_img), cmap='gray'); axes[3].set_title('Deteksi Tepi'); axes[3].axis('off')
        axes[4].imshow(cv2.cvtColor(metode.proses_hough(self.cv_img), cv2.COLOR_BGR2RGB)); axes[4].set_title('Hough'); axes[4].axis('off')
        axes[5].axis('off') # Kosongkan kotak terakhir

        if not os.path.exists('output'): os.makedirs('output')
        base_name = os.path.splitext(os.path.basename(self.img_path))[0]
        save_path = os.path.join('output', f'{base_name}_grid.jpg')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        messagebox.showinfo("Sukses", f"Hasil Grid berhasil disimpan di:\n{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiSegmentasi(root)
    root.mainloop()