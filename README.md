# LAB3: Convolution

### Рекомендации к выполнению лабораторной работы
1. Сделайте функцию `convolve(img, w)` -> `out` где `img.shape = (28, 28)`, `w.shape = (3, 3)`, `O.shape = (26, 26)`
   которая выполняет светрку 1 катринки с 1 фильтром

2. выполнем светрку для всех кортинок по всем фильтрам и каналам 
```
        for по батчам (B): 
            for по фильтрам (W):
                result_image = zeros()

                по каналам картинки (C):
                    result_image += `convolve`
                
                присваиваем result_image к общему результату
```