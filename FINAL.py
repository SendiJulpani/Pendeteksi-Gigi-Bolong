import sys #import library
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


# GUI untuk menampilkan gambar dan hasil klasifikasi daun uji
class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("GigiTambalan.ui", self)
        self.Image = None
        self.pushButton.clicked.connect(self.open)
        self.pushButton_3.clicked.connect(self.proses)
        self.pushButton_4.clicked.connect(self.crop)
        self.pushButton_5.clicked.connect(self.Contrass)
        self.pushButton_6.clicked.connect(self.sobel)
        self.pushButton_7.clicked.connect(self.threshold)
        self.pushButton_8.clicked.connect(self.filling)
        self.pushButton_9.clicked.connect(self.opening)

# -------------------------------------------------------------------------------------------------------
    def open(self): #Mendefinisikan sebuah metode bernama open.
        imagePath, _ = QFileDialog.getOpenFileName() #Membuka dialog file menggunakan QFileDialog untuk memilih file gambar. getOpenFileName akan mengembalikan path file yang dipilih dan nilai _ tidak digunakan.
        self.Image = cv2.imread(imagePath) #Membaca gambar menggunakan OpenCV (cv2.imread) dari imagePath yang telah dipilih. Gambar tersebut disimpan dalam atribut self.Image.
        pixmap = QPixmap(imagePath) #Membuat objek QPixmap dari imagePath yang akan digunakan untuk menampilkan gambar di antarmuka pengguna.
        self.label.setPixmap(pixmap) #Mengatur pixmap yang telah dibuat sebagai gambar pada label self.label.
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) #Mengatur posisi gambar pada label dengan menggabungkan konstanta AlignHCenter dan AlignVCenter. Ini akan mengatur posisi gambar secara horizontal dan vertikal tengah pada label.
        self.label.setScaledContents(True) #Mengatur konten label agar disesuaikan dengan ukuran label

# -------------------------------------------------------------------------------------------------------
    def threshold(self): #Mendefinisikan metode threshold.
        edges = self.Image #Mengambil gambar dari atribut self.Image dan menyimpannya ke variabel edges.
        # Thresholding untuk segmentasi
        threshold = 175 # Mengatur nilai threshold

        # Mengambil ukuran citra
        height, width = edges.shape[:2]

        # Melakukan thresholding secara manual
        #Membuat array thresholded dengan ukuran yang sama dengan edges
        thresholded = np.zeros_like(edges, dtype=np.uint8)
        for y in range(height): #Melakukan iterasi pada setiap piksel dalam gambar edges
            for x in range(width):
                # Jika nilai piksel tersebut melebihi nilai threshold
                if np.any(edges[y, x] > threshold):
                    #maka piksel pada array thresholded di posisi yang sama diatur menjadi 255 (putih)
                    thresholded[y, x] = 255
                else:
                    #Jika tidak, maka piksel tersebut diatur menjadi 0 (hitam).
                    thresholded[y, x] = 0

        #Menyimpan hasil thresholding dalam variabel self.Image
        self.Image = thresholded
        #menampilkan citra hasil thresholding pada label dengan menggunakan fungsi displayImage
        self.displayImage(2)

# -------------------------------------------------------------------------------------------------------
    def proses(self):
        image = self.Image

        # Konversi citra ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cropped_image = image
        # Menghitung nilai ambang batas
        threshold = 225

        # mendapatkan bentuk gambar
        height, width, channels = cropped_image.shape

        # buat gambar kosong dengan bentuk yang sama seperti cropped_image
        contrast_stretched = np.zeros_like(cropped_image)

        # loop melalui setiap piksel gambar
        for y in range(height):
            for x in range(width):
                # mendapatkan nilai piksel
                pixel_value = cropped_image[y, x]

                # periksa apakah nilai piksel kurang dari ambang batas
                if np.all(pixel_value < threshold):
                    # atur nilai piksel menjadi 0
                    contrast_stretched[y, x] = 0
                else:
                    # atur nilai piksel menjadi 255
                    contrast_stretched[y, x] = 255

        img = contrast_stretched.astype(np.uint8)
        # Mengatur nilai threshold
        threshold = 225

        # Mengambil ukuran citra
        height, width = img.shape[:2]

        # Melakukan thresholding secara manual
        contrast_stretched = np.zeros_like(gray, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if gray[y, x] > threshold:
                    contrast_stretched[y, x] = 255
                else:
                    contrast_stretched[y, x] = 0
        thresholded = contrast_stretched

        edges = thresholded
        # Operasi morfologi untuk filling holes
        # filled = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        # Buat kernel
        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]], dtype=np.uint8)

        # Mengatur nilai threshold
        threshold = 250

        # Mengambil ukuran citra
        height, width = edges.shape[:2]

        # Melakukan thresholding secara manual
        thresholded = np.zeros_like(edges, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if np.any(edges[y, x] > threshold):
                    thresholded[y, x] = 255
                else:
                    thresholded[y, x] = 0

        # Dapatkan dimensi citra
        height, width = thresholded.shape[:2]

        # Buat citra hasil
        filled = np.zeros_like(thresholded)

        # Operasi morfologi penutupan manual
        for i in range(height):
            for j in range(width):
                if np.all(thresholded[i, j] == 0):
                    continue
                filled[i - 1:i + 2, j - 1:j + 2] = 255

        nilai = filled

        # Buat kernel
        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]], dtype=np.uint8)

        # thresholded = np.where(filled > 127, 255, 0).astype(np.uint8)

        # Mengatur nilai threshold
        threshold = 200

        # Mengambil ukuran citra
        height, width = nilai.shape[:2]

        # Melakukan thresholding secara manual
        thresholded = np.zeros_like(nilai, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if np.any(nilai[y, x] > threshold):
                    thresholded[y, x] = 255
                else:
                    thresholded[y, x] = 0

        # Dapatkan dimensi citra
        height, width = thresholded.shape[:2]

        # Buat citra hasil
        opened = np.zeros_like(thresholded)

        # Operasi morfologi opening manual
        temp = np.zeros_like(thresholded)
        for i in range(height):
            for j in range(width):
                if np.all(thresholded[i, j] == 0):
                    continue
                temp[i - 1:i + 2, j - 1:j + 2] = 255

        for i in range(height):
            for j in range(width):
                if np.all(temp[i, j] == 255):
                    opened[i - 1:i + 2, j - 1:j + 2] = 255

        # Menentukan parameter untuk findContours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Membuat citra hasil pelabelan dengan tipe data uint8
        labeled_image = np.zeros_like(image, dtype=np.uint8)

        # Melakukan pelabelan
        for i, contour in enumerate(contours):
            # Membuat bounding box pada objek terdeteksi
            x, y, w, h = cv2.boundingRect(contour)

            # Menggambar bounding box dan memberikan label pada citra hasil
            cv2.rectangle(labeled_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(labeled_image, f"", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Menggabungkan citra asli dan labeled_image
        combined_image = cv2.bitwise_or(image, labeled_image)
        self.Image = combined_image
        self.displayImage(2)

    def sobel(self):
        contrast_stretched = self.Image
        # Deteksi tepi menggunakan operator Sobel
        # Inisialisasi kernel Sobel
        sobel_kernel_x = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])

        sobel_kernel_y = np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]])

        # Konvolusi dengan kernel Sobel untuk mengekstrak tepi horizontal
        # Membuat array dengan ukuran yang sama dengan contrast_stretched yang diinisialisasi dengan nilai 0 menggunakan np.zeros_like.
        edges_x = np.zeros_like(contrast_stretched)
        for i in range(1, contrast_stretched.shape[0] - 1):
            for j in range(1, contrast_stretched.shape[1] - 1):
                gx = np.sum(sobel_kernel_x * contrast_stretched[i - 1:i + 2, j - 1:j + 2])
                edges_x[i, j] = gx

        # Konvolusi dengan kernel Sobel untuk mengekstrak tepi vertikal
        edges_y = np.zeros_like(contrast_stretched)
        for i in range(1, contrast_stretched.shape[0] - 1):
            for j in range(1, contrast_stretched.shape[1] - 1):
                gy = np.sum(sobel_kernel_y * contrast_stretched[i - 1:i + 2, j - 1:j + 2])
                edges_y[i, j] = gy

        # Menggabungkan hasil deteksi tepi secara horizontal dan vertikal
        edges = np.abs(edges_x) + np.abs(edges_y)

        # Konversi ke citra dengan tipe data uint8
        edges = np.uint8(np.clip(edges, 0, 255))

        self.Image = edges
        self.displayImage(2)

    def crop(self): #Mendefinisikan metode crop.

        image = self.Image #Mengambil gambar dari atribut self.Image
        rows, cols, channels = image.shape[:3] #mendapatkan jumlah baris (rows), jumlah kolom (cols), dan jumlah saluran (channels) dari gambar

        #Menentukan titik awal dan ukuran potongan yang akan dipotong dari gambar
        x = 80
        y = 30
        width = 400
        height = 400

        #Mengatur koordinat x dan y untuk memastikan bahwa potongan tidak keluar dari batas gambar. Jika nilai x atau y di luar batas
        x = min(max(0, x), cols - 1)
        y = min(max(0, y), rows - 1)
        width = min(width, cols - x)
        height = min(height, rows - y)

        #Membuat array dengan ukuran yang sesuai dengan potongan menggunakan np.zeros.
        cropped_image = np.zeros((height, width, channels), dtype=np.uint8)

        #Melakukan iterasi pada setiap piksel dalam potongan gambar dan mengambil piksel yang sesuai dari gambar asli.
        for row in range(height):
            for col in range(width):

                pixel = image[y + row, x + col]

                cropped_image[row, col] = pixel

        # Menentukan nilai piksel minimum dan maksimum
        min_val = np.min(cropped_image)
        max_val = np.max(cropped_image)

        self.Image = cropped_image
        self.displayImage(2)

    # def grayscale(self):  # fungsi untuk mengubah citra ke grayscale
    #     H, W = self.Image.shape[:2]
    #     gray = np.zeros((H, W), np.uint8)
    #     for i in range(H):
    #         for j in range(W):
    #             gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] + 0.587
    #                                  * self.Image[i, j, 1] + 0.114 * self.Image[i, j, 2], 0, 255)
    #     self.Image = gray
    #     self.displayImage(2)

    def Contrass(self):  #fungsi untuk mengatur kontras

        image = self.Image # Mengambil gambar dari atribut self.Image
        threshold = 225 # Nilai ambang batas untuk kontras


        height, width, channels = image.shape[:3] # Mendapatkan dimensi gambar

        contrast_stretched = np.zeros_like(image)  # Membuat array kosong dengan ukuran yang sama dengan gambar



        for y in range(height):
            for x in range(width):

                pixel_value = image[y, x] # Mendapatkan nilai piksel pada koordinat (y, x)


                if np.all(pixel_value < threshold): # Jika nilai piksel lebih kecil dari threshold

                    contrast_stretched[y, x] = 0 # Set nilai piksel pada array kontras ke 0
                else:

                    contrast_stretched[y, x] = 255 # Set nilai piksel pada array kontras ke 255

        self.Image = contrast_stretched  # Menyimpan gambar hasil kontras pada atribut self.Image
        self.displayImage(2) # Menampilkan gambar hasil kontras dengan memanggil fungsi displayImage() dengan argumen 2

    def filling(self):
        edges = self.Image
        # Operasi morfologi untuk filling holes
        # filled = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

        # Buat kernel
        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]], dtype=np.uint8)

        # Mengatur nilai threshold
        threshold = 250

        # Mengambil ukuran citra
        height, width = edges.shape[:2]

        # Melakukan thresholding secara manual
        thresholded = np.zeros_like(edges, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if np.any(edges[y, x] > threshold):
                    thresholded[y, x] = 255
                else:
                    thresholded[y, x] = 0

        # Dapatkan dimensi citra
        height, width = thresholded.shape[:2]

        # Buat citra hasil
        filled = np.zeros_like(thresholded)

        # Operasi morfologi penutupan manual
        for i in range(height):
            for j in range(width):
                if np.all(thresholded[i, j] == 0):
                    continue
                filled[i - 1:i + 2, j - 1:j + 2] = 255
        self.Image = filled
        self.displayImage(2)

    def opening(self):
        nilai = self.Image

        # Buat kernel
        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]], dtype=np.uint8)

        # thresholded = np.where(filled > 127, 255, 0).astype(np.uint8)

        # Mengatur nilai threshold
        threshold = 200

        # Mengambil ukuran citra
        height, width = nilai.shape[:2]

        # Melakukan thresholding secara manual
        thresholded = np.zeros_like(nilai, dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if np.any(nilai[y, x] > threshold):
                    thresholded[y, x] = 255
                else:
                    thresholded[y, x] = 0

        # Dapatkan dimensi citra
        height, width = thresholded.shape[:2]

        # Buat citra hasil
        opened = np.zeros_like(thresholded)

        # Operasi morfologi opening manual
        temp = np.zeros_like(thresholded) #Membuat array temp dengan ukuran yang sama dengan thresholded yang diinisialisasi dengan nilai 0
        for i in range(height): #Melakukan iterasi pada setiap piksel dalam gambar thresholded.
            for j in range(width):
                if np.all(thresholded[i, j] == 0): #Jika piksel tersebut memiliki nilai bukan 0
                    continue
                # maka dilakukan penyebaran nilai 255 ke piksel-piksel sekitarnya pada array temp.
                temp[i - 1:i + 2, j - 1:j + 2] = 255

        for i in range(height):
            for j in range(width):
                #Jika piksel tersebut memiliki nilai 255
                if np.all(temp[i, j] == 255):
                    # maka dilakukan penyebaran nilai 255 ke piksel-piksel sekitarnya pada array opened
                    opened[i - 1:i + 2, j - 1:j + 2] = 255
        self.Image = opened
        self.displayImage(2)

    # Mendefinisikan metode displayImage yang menerima argumen window.
    def displayImage(self, window):
        # Menginisialisasi variabel qformat dengan nilai awal QImage.Format_Indexed8.
        qformat = QImage.Format_Indexed8

        # Memeriksa apakah gambar memiliki tiga dimensi, yang menandakan gambar berwarna.
        if len(self.Image.shape) == 3:
            # Jika gambar memiliki tiga dimensi dan dimensi ketiga memiliki nilai 4
            if (self.Image.shape[2]) == 4:
                # maka format gambar ditetapkan sebagai QImage.Format_RGBA8888
                qformat = QImage.Format_RGBA8888
            else:
                # Jika tidak, format gambar ditetapkan sebagai QImage.Format_RGB888.
                qformat = QImage.Format_RGB888

        #Membuat objek QImage dari self.Image
        img = QImage(
            #dengan menggunakan ukuran dan format gambar yang telah ditentukan sebelumnya.
            self.Image,
            self.Image.shape[1],
            self.Image.shape[0],
            self.Image.strides[0],
            qformat,
        )

        img = img.rgbSwapped() #Mengganti komponen warna gambar menjadi urutan yang benar.

        if window == 1: #Mengecek apakah nilai parameter window adalah 1
            #Jika benar, maka citra akan ditampilkan pada label dengan nama label_Load.
            self.label_Load.setPixmap(QPixmap.fromImage(img))
            #Mengatur pixmap citra pada label label_Load menggunakan objek QPixmap yang dibuat dari objek QImage img.
            self.label_Load.setAlignment(
                #Mengatur alignment (penyusunan) teks pada label secara horizontal dan vertikal menjadi tengah.
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
            )
            #Mengatur agar konten pada label akan diubah ukurannya secara proporsional untuk mengisi seluruh area label.
            self.label_Load.setScaledContents(True)
        elif window == 2: #Jika window bukan 1
            # Jika benar, citra akan ditampilkan pada label dengan nama label_2.
            self.label_2.setPixmap(QPixmap.fromImage(img))
            #Mengatur alignment (penyusunan) teks pada label secara horizontal dan vertikal menjadi tengah.
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)
        print("----Nilai Pixel Citra----\n", self.Image)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle("Project Akhir")
window.show()
sys.exit(app.exec_())