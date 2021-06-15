# AiPO-Project

## Instalacja
* Po ściągnieciu repozytorium należy pobrać plik z wagami potrzebnymi do działania programu ze strony https://pjreddie.com/media/files/yolov3.weights a następnie przenieść go do folderu **/yolov3-coco**. Testowo w aplikacji znajdują się dwa filmy w folderze test_videos, jeżeli chcemy użyć innego filmu należy zmienić zmienną **VIDEO_PATH** znajdującą się u góry pliku **people_counting.py**
## Użyte rozwiązanie
### YOLOv3
W naszym projekcie zastosowaliśmy system detekcji obiektów w czasie rzeczywistym YOLOv3. YOLO traktuje wykrywanie obiektów na obrazie jako pojedynczy problem
regresji, stąd też jego nazwa, ponieważ tylko raz analizuje cały obraz, co znacznie przyspiesza cały proces detekcji obiektów. Na poniższym obrazie przedstawiony został sposób jego działania. ![Screenshot 2021-06-15 at 21 03 29](https://user-images.githubusercontent.com/61520943/122108992-25e3ed80-ce1d-11eb-8f41-e280d965ca73.png)

Do wykrywania postaci użyliśmy wag dostarczanonych przez twórców systemu 'yolov3.weights'. 

### Sposób działania aplikacji
System analizuje wideo klatka po klatce wykorzystując bibliotekę OpenCV, następnie każda klatka jest przetwarzana przez model YOLO. Zwraca on wykryte obiekty podpisane odpowiednimi etykietami a następnie filtrujemy je wybierając te, które oznaczone są jako osoba. Jeżeli pewność osiąga określony próg rysujemy wówczas prostokąt wokół tej osoby oraz zliczamy ilość obiektów na danej klatce. 

