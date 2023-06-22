# Урок 10. Распознавание лиц и эмоций
# Задание по итогам курса:
# (упрощенное/для тех, у кого нет вебкамеры)
#
# Нужно написать приложение, которое будет получать на вход изображение.
# В процессе определять, что перед камерой находится человек, задетектировав его лицо на кадре.
# На изображении человек показывает жесты руками, а алгоритм должен считать их и классифицировать.
# (более сложное)
#
# Нужно написать приложение, которое будет считывать и выводить кадры с веб-камеры.
# В процессе считывания определять что перед камерой находится человек, задетектировав его лицо на кадре.
# Человек показывает жесты руками, а алгоритм должен считать их и классифицировать.
# Для распознавания жестов, вам надо будет скачать датасет https://www.kaggle.com/gti-upm/leapgestrecog,
# разработать модель для обучения и обучить эту модель.

# *(Усложненное задание) Все тоже самое, но воспользоваться этим датасетом: https://fitnessallyapp.com/datasets/jester/v1

# Как работать с веб-камерой на google colab https://stackoverflow.com/questions/54389727/opening-web-camera-in-google-colab
# У кого нет возможности работать через каггл (нет верификации), то можете данные взять по ссылке: https://disk.yandex.ru/d/R2PGlaXDf6_HzQ


import classes

handler = classes.GestureHandler()
streamer = classes.VideoStreamer()
streamer.run(handler)
print('All done.')