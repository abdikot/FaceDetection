import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml") # meload file 'face_ref.xml' dan memassukanya ke dalam variabel face_ref
camera = cv2.VideoCapture(0) # (0) digunakan untuk mengakses kamera bawaan devices 

# function yang akan digunakan untuk mendeteksi wajah
def face_detect(frame):
    optimize_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # membuat gambar menjadi hitam putih 'grayscale'
    # detectMultiScale  = digunakan untuk mendeteksi lebih dari 1 object wajah
    # scaleFactor       =  faktor skala pada setiap frame untuk melakukan perubahan ukuran gambar
    faces = face_ref.detectMultiScale(optimize_frame, scaleFactor=1.1) # variabel 'faces' yang digunakan untuk menampung nilai dari wajah yang terdeteksi
    return faces

# function yang digunakan untuk menampilkan 'box' pada wajah yang terdeteksi
def drawer_box(frame):

    # x dan w = lebar(horizontal)
    # y dan h = tinggi(vertical)
    for x,y,w,h in face_detect(frame):
        #             img,       x dan y,   perhitungan antara x y w h,     warna(B,G,R),      ketebalan
        cv2.rectangle(frame,     (x, y),    (x + w, y + h),                 (0,255,0)),        4

# function untuk menutup semua  resource yang dipakai oleh program
def close_window():
    camera.release() # menutup kamera 
    cv2.destroyAllWindows() # menutup semua 'window' yang dijalankan program 
    exit() # mematikan program

# function utama yang berfungsi sebagai loop  utama program
def main():
    while True:
        _, frame = camera.read() # mengakses kamera
        drawer_box(frame) # memanggil function 'drawer_box'
        cv2.imshow("Face detection", frame) # menampilkan window dengan nama 'Face detection' dan 'frame' dari kamera
        if  cv2.waitKey(1) & 0xFF == ord('q'): # if yang akan menutup program jika tombol 'q' ditekan
            close_window() # memanggil function  'close_window'

if __name__=="__main__":
    main()
