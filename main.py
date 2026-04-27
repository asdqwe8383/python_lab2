# Лабораторная работа №2
# Вариант 21
# Визуализация и спектральный анализ речевого сигнала

import time
import wave

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FILE_NAME = "21.wav"


# Запоминаем время начала работы программы.
start_time = time.time()

print("Лабораторная работа №2")
print("Вариант 21")
print("Программа выполняет визуализацию и спектральный анализ wav-файла.")
print("Для варианта 21 требуется wav-файл с частотой дискретизации 32000 Гц.")
print("Файл по умолчанию:", DEFAULT_FILE_NAME)
print()


def is_wav_file_name(file_name):
    """Проверяет, что имя файла заканчивается на .wav."""
    return file_name.lower().endswith(".wav")


# Ввод имени файла.
# Если пользователь просто нажмет Enter, будет использован файл 21.wav.
file_name = input(
    "Введите имя wav-файла или нажмите Enter для 21.wav: "
).strip()

if file_name == "":
    file_name = DEFAULT_FILE_NAME
elif not is_wav_file_name(file_name):
    print("Имя файла должно заканчиваться на .wav.")
    print("Будет использовано имя файла по умолчанию:", DEFAULT_FILE_NAME)
    file_name = DEFAULT_FILE_NAME


# Количество отсчетов нужно для первого графика.
while True:
    try:
        samples_count = int(
            input("Введите количество отсчетов для первого графика: ")
        )

        if samples_count > 0:
            break

        print("Ошибка: количество отсчетов должно быть положительным.")
    except ValueError:
        print("Ошибка: введите целое число.")


# Открываем wav-файл и считываем его параметры.
try:
    with wave.open(file_name, "rb") as wav_file:
        channels_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames_count = wav_file.getnframes()

        frames = wav_file.readframes(frames_count)

except FileNotFoundError:
    print("\nОшибка: файл не найден.")
    print("Проверьте, что wav-файл находится в одной папке с программой.")
    print(time.time() - start_time, "seconds")
    raise SystemExit

except wave.Error:
    print("\nОшибка: файл не является корректным wav-файлом.")
    print(time.time() - start_time, "seconds")
    raise SystemExit


print("\nИнформация о wav-файле:")
print("Количество каналов:", channels_count)
print("Размер отсчета в байтах:", sample_width)
print("Частота дискретизации:", sample_rate, "Гц")
print("Количество отсчетов:", frames_count)


# По варианту 21 нужна частота дискретизации 32000 Гц.
# При другой частоте программа не падает, а продолжает работу.
if sample_rate != 32000:
    print("\nПредупреждение:")
    print("Для варианта 21 требуется частота дискретизации 32000 Гц.")
    print("Текущий файл имеет частоту", sample_rate, "Гц.")
    print("Программа продолжит работу с фактической частотой файла.")


# По заданию нужен mono-файл.
# Если каналов несколько, анализируем первый канал.
if channels_count != 1:
    print("\nПредупреждение:")
    print("По заданию нужен mono-файл.")
    print("Текущий файл содержит", channels_count, "канала.")
    print("Для анализа будет использован первый канал.")


# Для PCM signed 16 bit один отсчет занимает 2 байта.
if sample_width != 2:
    print("\nОшибка:")
    print("Программа рассчитана на wav-файлы PCM signed 16 bit.")
    print("В данном файле размер отсчета равен", sample_width, "байт.")
    print(time.time() - start_time, "seconds")
    raise SystemExit


# Преобразуем байты wav-файла в массив числовых отсчетов.
signal = np.frombuffer(frames, dtype=np.int16)


# Если файл многоканальный, оставляем только первый канал.
if channels_count > 1:
    signal = signal.reshape(-1, channels_count)
    signal = signal[:, 0]


# Если пользователь запросил больше отсчетов, чем есть в файле,
# строим все доступные отсчеты.
if samples_count > len(signal):
    print("\nПредупреждение:")
    print("В файле меньше отсчетов, чем было запрошено.")
    print("Будут построены все доступные", len(signal), "отсчетов.")
    samples_count = len(signal)


# Номера отсчетов для первого графика.
sample_numbers = np.arange(samples_count)

# Время для осциллограммы всего сигнала.
time_values = np.arange(len(signal)) / sample_rate


# Выполняем дискретное преобразование Фурье.
# Результат содержит комплексные числа Re + j * Im.
spectrum = np.fft.fft(signal)


# Получаем частоты для значений ДПФ.
frequencies = np.fft.fftfreq(len(signal), d=1 / sample_rate)


# Для отображения оставляем только положительную половину спектра.
positive_frequencies = frequencies[:len(frequencies) // 2]
positive_spectrum = spectrum[:len(spectrum) // 2]


# Для варианта 21 требуется модуль ДПФ:
# sqrt(Re**2 + Im**2).
spectrum_module = np.abs(positive_spectrum)


# Создаем окно с четырьмя графиками.
plt.figure(figsize=(14, 10))


# 1. Линейный сплошной график заданного количества отсчетов.
plt.subplot(2, 2, 1)
plt.plot(sample_numbers, signal[:samples_count], linestyle="-")
plt.title("Первые отсчеты звукового сигнала")
plt.xlabel("Номер отсчета")
plt.ylabel("Амплитуда, у.е.")
plt.grid(True)


# 2. Осциллограмма всего сигнала как функция времени.
plt.subplot(2, 2, 2)
plt.plot(time_values, signal, linestyle="-")
plt.title("Осциллограмма звукового сигнала")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда, у.е.")
plt.grid(True)


# 3. Спектральный анализ: модуль ДПФ.
plt.subplot(2, 2, 3)
plt.plot(positive_frequencies, spectrum_module, linestyle="-")
plt.title("Спектральный анализ: модуль ДПФ")
plt.xlabel("Частота, Гц")
plt.ylabel("Модуль ДПФ, у.е.")
plt.grid(True)


# 4. Гистограмма отсчетов звукового сигнала.
plt.subplot(2, 2, 4)
plt.hist(signal, bins=50)
plt.title("Гистограмма отсчетов звукового сигнала")
plt.xlabel("Амплитудный интервал, у.е.")
plt.ylabel("Количество отсчетов")
plt.grid(True)


plt.tight_layout()
print(time.time() - start_time, "seconds")
plt.show()


print("\nПрограмма успешно завершена.")
