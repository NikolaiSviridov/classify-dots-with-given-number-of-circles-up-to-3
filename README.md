# Классификация точек при заданном количестве окружностей
## Файлы
1. 
## Описание Алгоритма
Алгоритм разделен на 2 части, которые обрабатывают данные с заданными ограничениями. 

## 1 часть алгоритма
Алгоритм работает на основе принципа Дирихле и построении окружности на основе 3 точек.  
Главной идеей является нахождение $M - 1$ истинной окружности, основываясь на которых, мы можем классифицировать остальные точки.  
На примере:  
1. Вход даны 3 окружности
2. Алгоритм найдет 1 истинную окружность и сведет задачу к задаче 2 окружностей, классифицировав и убрав точки относящиеся к найденной окружности
3. Находит последнюю истинную окружность
4. На основе найденных окружностей классифицирует точки, а оставшиеся принадлежат не найденной
### Случай 3 окружностей
По принципу Дирихле нам понадобится 7 точек, чтобы 3 из них лежали на одной из окружностей. 
1. Выбираем 7 точек получаем все возможные окружности, которые мы можем построить на 3 точках. 
2. Далее проходим по всем точкам и проверяем, принадлежат ли они найденным окружностям
3. Если одна из окружностей содержит $\geq$  7  точек, то мы нашли окружность

Пункт 3 обусловлен тем, что у нас возможна ситуация, когда попалась окружность, которая лежит на 
реальных окружностях. Такая окружность может содержать в себе 6 точек (по 2 с каждой окружности).
Данный алгоритм запускается максимум 3 раза и отработает на $\geq$ 21 точке (подробнее описание в 'Худшие случаи').

### Случай 2 окружностей
Аналогично случаю с 3 окружностями. Вместо 7 выбираем 5 точек, а отработает он максимум 2 раза 
на $\geq$ 10 точках.

### Худшие случаи
Все худшие случаи основаны на том, что мы можем неудачно выбирать 3 точки, т.е. они не будут находиться в истинной окружности или окружность будет иметь меньше необходимого количества точек.  
Здесь '+' означает, что число может быть больше. Пример: 6+ (это число $\geq$ 6)

#### 3 круга
Рассмотрим ситуацию, когда у нас имеется 3 круга $C_1$, $C_2$, $C_3$.  
Точки распределены следующим образом:
| $C_1$ | $C_2$ | $C_3$ |
|-------|-------|-------|
| 7+    | 6     | 6     |

Худший случай, когда у нас из 7 точек только 2 точки лежат в подходящей окружности, а остальные по 2 и 3 в оставшихся (т.е. отработает в пустую, т.к. окружности не наберут 5 точек ). 
Таблица итераций:  


| итерация | значение | $C_1$ | $C_2$ | $C_3$ |
|:--------:|:--------:|:-----:|:-----:|:-----:|
|    1     | исходное |  7+   |   6   |   6   |
|          | выбрали  |   2   |   3   |   2   |
|    2     | исходное |  5+   |   3   |   4   |
|          | выбрали  |   2   |   2   |   3   |
|    3     | исходное |  3+   |   1   |   1   |
|          | выбрали  |   5   |   1   |   1   |

Т.е. получается, для того что-бы работал алгоритм нужно минимум 21 точка и максимум 3 итерации.

#### 2 круга
Рассмотрим ситуацию, когда у нас имеется 2 круга $C_1$, $C_2$.  
Точки распределены следующим образом:
| $C_1$ | $C_2$ |
|-------|-------|
| 5+    | 4     |
  
Худший случай, когда у нас из 5 точек 2 лежат в $C_1$, а остальные в $C_2$ (т.е. отработает в пустую, т.к. окружности не наберут 5 точек ). 
Таблица итераций:

| итерация | значение | $C_1$ | $C_2$ |
|:--------:|:--------:|:-----:|:-----:|
|    1     | исходное |  5+   |   4   |
|          | выбрали  |   2   |   3   |
|    2     | исходное |  3+   |   1   |
|          | выбрали  |   4   |   1   |

Т.е. получается, для того что-бы работал алгоритм нужно минимум 10 точек и максимум 2 итерации.

## 2 часть алгоритма
Эта часть алгоритма начинает работать, если данные не удовлетворяют ограничениям 1 части. Также из-за малого количества точек результат не детерминирован.
### 3 окружности
1. Для случая с 3 или 4 точками они классифицируются случайно, т.к. любая содержит 1 точку и также любая может содержать 2
2. Если точек больше, ищем окружность включающую, как можно больше точек либо 7 (7 т.к. это 100% истинная окружность)
3. Классифицируем точки из максимальной окружности, переходим к решению задачи для 2 окружностей
### 2 окружности
1. Для случая с 2 или 3 точками они классифицируются случайно, т.к. любая содержит 1 точку и также любая может содержать 2
2. Если точек больше, ищем окружность включающую, как можно больше точек либо 5 (5 т.к. это 100% истинная окружность)
3. Классифицируем точки из максимальной окружности, выводим результат
## Требования

1. numpy
2. scipy
3. matplotlib (если рассматриваете nootebook)
4. math
