## Самигуллин Равиль гр 6133

# Pycuda
Данную лабораторную я так же реализовал с помощью pycuda.

Реализация параллели:

```
calculate_median_GPU = SourceModule("""
    texture<unsigned int, 2, cudaReadModeElementType> tex;

    __global__ void median_gpu(unsigned int * __restrict__ d_result, const int M, const int N)
    {
        const int i = threadIdx.x + blockDim.x * blockIdx.x;
        const int j = threadIdx.y + blockDim.y * blockIdx.y;

        if ((i >= 2) && (i < M-1) && (j >= 2) && (j < N-1)) {
            float t[9];
            int index = 0;

            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    t[index] = tex2D(tex, j+dj, i+di);
                    index++;
                }
            }

            for (int k = 0; k < 8; k++) {
                for (int l = 0; l < 8-k; l++) {
                    if (t[l] > t[l+1]) {
                        float temp = t[l];
                        t[l] = t[l+1];
                        t[l+1] = temp;
                    }
                }
            }

            d_result[i * N + j] = t[4];
        }
    }
    """)
```
```
tex = calculate_median_GPU.get_texref("tex")
tex.set_filter_mode(driver.filter_mode.LINEAR)
tex.set_address_mode(0, driver.address_mode.MIRROR)
tex.set_address_mode(1, driver.address_mode.MIRROR)
driver.matrix_to_texref(image.astype(np.uint32), tex, order="C")
start = time()
median_gpu(driver.Out(gpu_result), np.int32(M), np.int32(N), block=block, grid=grid, texrefs=[tex])
```

1. Определение текстурирующего объекта `tex`, который предоставляет доступ к исходному изображению в формате текстуры на GPU.
2. Объявление функции `median_gpu`, которая будет выполняться на каждом потоке GPU. Эта функция получает текущие индексы пикселя `(i, j)` в изображении и использует двойной цикл для итерации по окрестности размером 3x3 вокруг данного пикселя.
3. Значения пикселей окрестности считываются через текстурирующий объект `tex` с использованием функции `tex2D`.
4. Значения пикселей окрестности сохраняются в массив `t`.
5. Затем выполняется сортировка значений массива `t` для нахождения медианного значения. Для этого используется метод пузырьковой сортировки.
6. Медианное значение вычисленной окрестности сохраняется в соответствующий пиксель результирующего изображения `gpu_result`.
7. Функция `driver.matrix_to_texref` используется для загрузки исходного изображения в текстуру `tex`, чтобы она была доступна внутри ядра `median_gpu`.

# Результаты экспериментов

Провел фильтрацию с использованием GPU И CPU на изображениях разного разрешения. Сравнил ускорение. 

Изображения 650x427

<image src="pictures/640x427.png" alt="Зашумленное изображение"  width="330"/> <image src="pictures/640x427-noise(1).png" alt="Зашумленное изображение"  width="330"/> <image src="pictures/640x427-очищенное gpu.png" alt="Зашумленное изображение"  width="330"/> 

Изображения 1280x720

<image src="pictures/1280x720.png" alt="Зашумленное изображение"  width="330"/> <image src="pictures/1280x720-noise.png" alt="Зашумленное изображение"  width="330"/> <image src="pictures/1280x720 Очищенное gpu.png" alt="Зашумленное изображение"  width="330"/> 

Изображения 1280x1920

<image src="pictures/1280x1920.png" alt="Зашумленное изображение"  width="330"/> <image src="pictures/1280x1920-noise.png" alt="Зашумленное изображение"  width="330"/> <image src="pictures/1280x1920 Очищенное gpu.png" alt="Зашумленное изображение"  width="330"/> 

Изображения 4096x2560

<image src="pictures/4096x2560.png" alt="Зашумленное изображение"  width="330"/> <image src="pictures/4096x2560 -noise.png" alt="Зашумленное изображение"  width="330"/> <image src="pictures/4096x2560 -очищенное gpu.png" alt="Зашумленное изображение"  width="330"/> 

Таблица результатов

|Разрешение изображения|640x427|1280x720|1920x1280|4096x2560|
| :- | :- | :- | :- | :- |
|CPU\_Time|9873 ms.| 24806 ms.|51580 ms.| 266468 ms. |
|GPU\_Time|0.9039 ms.|1.9 ms.|3.1 ms.| 38.1 ms. |
|Ускорение| 9969 | 12984 | 16323 | 6984.95 |

ms. - milliseconds

# Заключение

Выводы: Я реализовал medial filter на CPU и GPU, провел анализ для 4 изображений с разным разрешением, подсчитал время выполнения и ускорение, заметно что ускорение так же растет, при увеличении разрешения картинки, что логично объясняет сложность обработки больших массивов для cpu.
