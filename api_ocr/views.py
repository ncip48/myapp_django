from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions

import os
import shutil
import uuid
from typing import Annotated

# OCR Stater
import easyocr
import cv2
import numpy as np
import time
import json

from urllib.request import urlopen
def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image


def generate_random_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    random_name = str(uuid.uuid4())
    return f"{random_name}.{ext}"

def generate_random_folder():
    random_dir = os.path.join('image', str(uuid.uuid4()))
    return random_dir

# Fungsi untuk memuat dan mengoptimalkan gambar
def load_and_preprocess_image(image_path):
    image = image_path
    # Misalnya, mengubah gambar menjadi skala abu-abu untuk mengurangi ukuran dan kompleksitas
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bisa juga dilakukan proses prapemrosesan lainnya, seperti pengurangan noise atau peningkatan kontras
    return gray_image

# Convert NumPy int64 to regular Python integers
def convert_np_int64(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    return obj

# Fungsi untuk melakukan OCR pada gambar yang telah diolah
def perform_ocr(image):
    # Membuat objek Reader dengan dukungan GPU
    reader = easyocr.Reader(['en'], gpu=True)
    # Membaca teks dari gambar
    start_time = time.time()
    results = reader.readtext(image)
    end_time = time.time()
    # Menghitung waktu yang dibutuhkan
    processing_time_ms = (end_time - start_time) * 1000
    return results, processing_time_ms

def get_name(hasil):
    try:
        index_nama = hasil[int(hasil.index("Nama")) + 1]
        return index_nama
    except ValueError:
        return None

def get_gender(hasil):
    try:
        index_nama = hasil[int(hasil.index("Jenis Kelamin")) + 1]
        return index_nama
    except ValueError:
        return None

def get_blood(hasil):
    try:
        index_nama = hasil[int(hasil.index("Gol. Darah")) + 1]
        return index_nama
    except ValueError:
        return None


def get_address(hasil):
    try:
        index_nama = hasil[int(hasil.index("Alamat")) + 1]
        return index_nama
    except ValueError:
        return None

def get_rtrw(hasil):
    try:
        index_nama = hasil[int(hasil.index("RTIRW")) + 1]
        return index_nama
    except ValueError:
        return None

def get_desa(hasil):
    try:
        index_nama = hasil[int(hasil.index("KellDesa")) + 1]
        return index_nama
    except ValueError:
        return None

def get_kecamatan(hasil):
    try:
        index_nama = hasil[int(hasil.index("Kecamatan")) + 1]
        return index_nama
    except ValueError:
        return None

def get_religion(hasil):
    try:
        index_nama = hasil[int(hasil.index("Agama")) + 1]
        return index_nama
    except ValueError:
        return None

def get_kawin(hasil):
    try:
        index_nama = hasil[int(hasil.index("Status Perkawinan")) + 1]
        return index_nama
    except ValueError:
        return None

def get_work(hasil):
    try:
        index_nama = hasil[int(hasil.index("Pekerjaan")) + 1]
        return index_nama
    except ValueError:
        return None

def get_warga(hasil):
    try:
        index_nama = hasil[int(hasil.index("Kewarganegaraan")) + 1]
        return index_nama
    except ValueError:
        return None

class OcrApiView(APIView):
    # add permission to check if user is authenticated
    # permission_classes = [permissions.IsAuthenticated]

    # 2. Create
    def post(self, request, *args, **kwargs):
        image_form = request.data.get('image')

        valid_results = 0
        hasil = []

        upload_dir = generate_random_folder()
        os.makedirs(upload_dir, exist_ok=True)

        image_cv2 = url_to_image(image_form)

        # Memuat dan mengoptimalkan gambar
        processed_image = load_and_preprocess_image(url_to_image(image_form))

        # Melakukan OCR pada gambar yang telah diolah
        results, processing_time_ms = perform_ocr(processed_image)

        # Memfilter hasil OCR yang hanya mengandung angka dan panjangnya 16 digit
        for result in results:
            text = result[1]
            hasil.append(text)

            # Menghapus spasi dan karakter non-angka
            digit_text = ''.join(filter(str.isdigit, text))
            # Memeriksa apakah panjangnya 16 digit
            if len(digit_text) == 16:
                valid_results = digit_text

        response = {
            "success": True,
            "message": "Success Get Data KTP",
            "data": {
                'time_processing': processing_time_ms / 1000,
                'province' : hasil[0],
                'city' : hasil[1],
                'nik': str(valid_results),
                'nama_lengkap': get_name(hasil),
                'jenis_kelamin' : get_gender(hasil),
                'golongan_darah' : get_blood(hasil),
                'alamat' : get_address(hasil),
                'rtrw' : get_rtrw(hasil),
                'desa' : get_desa(hasil),
                'kecamatan' : get_kecamatan(hasil),
                'agama' : get_religion(hasil),
                'status_kawin' : get_kawin(hasil),
                'pekerjaan' : get_work(hasil),
                'kewarganegaraan' : get_warga(hasil),
            }
        }

        return Response(response)