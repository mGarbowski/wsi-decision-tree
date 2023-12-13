# Sprawozdanie
Mikołaj Garbowski

## Zadanie
Przedmiotem zadania była implementacja algorytmu ID3 i ewaluacja jego działania na zbiorach danych
* [Breast Cancer](https://archive.ics.uci.edu/dataset/14/breast+cancer)
* [Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)

Wyniki eksperymentów przedstawiają średnie z 25 wykonań algorytmu przy losowym podziale na zbiór uczący i testujący.
Wszystkie eksperymenty były wykonane przy proporcjach 3:2 wielkości zbioru uczącego do testującego (chyba że napisane inaczej).

## Wyniki dla zbioru breast cancer
```
Average values over 25 runs on data/breast+cancer/breast-cancer.data dataset
Number of samples in test set: 115
Accuracy:    65.88%
Precision:   73.74%
Recall:      79.69%
Specificity: 34.51%

TP=64     FN=16    
FP=23     TN=12    
```


## Wyniki dla zbioru mushroom
```
Average values over 25 runs on data/mushroom/agaricus-lepiota.data dataset
Number of samples in test set: 3250
Accuracy:    99.99%
Precision:   99.99%
Recall:      99.99%
Specificity: 99.99%

TP=1677   FN=0     
FP=0      TN=1572  
```

## Hipotezy

### 1. Brakujące wartości atrybutów
Ok. 3% rekordów w zbiorze breast cancer ma brakujące wartości (oznaczone `?`).
Wyniki dla zbioru breast cancer z wyrzuconymi wierszami zawierającymi puste wartości są bardzo zbliżone do tych bez modyfikacji

```
Average values over 25 runs on breast cancer without missing values dataset
Number of samples in test set: 111
Accuracy:    65.73%
Precision:   73.88%
Recall:      79.07%
Specificity: 35.59%

TP=61     FN=16    
FP=22     TN=12 
```

### 2. Przeuczenie modelu
Model oparty na algorytmie ID3 może być przeuczony do zbioru uczącego.
Wykonałem eksperyment dla proporcji 1:4 i 2:3 wielkości zbioru uczącego do testującego.
W obu przypadkach wyniki są zbliżone, minimalnie słabsze niż w pierwszym eksperymencie.

#### Proporcje 1:4
```
Average values over 25 runs on breast cancer dataset
Number of samples in test set: 229
Accuracy:    63.35%
Precision:   72.16%
Recall:      77.73%
Specificity: 30.02%

TP=124    FN=36    
FP=48     TN=21 
```

#### Proporcje 2:3
```
Number of samples in test set: 172
Accuracy:    63.65%
Precision:   73.11%
Recall:      76.28%
Specificity: 34.83%

TP=92     FN=29    
FP=34     TN=18 
```

### 3. Grupowanie wartości liczbowe
Zbiór zawiera atrybuty, które są de facto liczbowe (wiek, rozmiar nowotworu), 
ale w zbiorze są pogrupowane w przedziały i traktowane jako kategorie (np. 20-29).

Hipoteza: można znaleźć lepsze punkty podziału.

Nie ma możliwości łatwego sprawdzenia tej hipotezy, ponieważ pierwotne wartości liczbowe nie są podane w zbiorze.

## Wnioski
Sprawdzane hipotezy nie wyjaśniają rozbieżności w skuteczności klasyfikatora między badanymi zbiorami danych.
Wyniki uzyskane przez mój model na zbiorze breast cancer są zbliżone do dolnych granic widełek dokładności podanych na diagramie
na stronie https://archive.ics.uci.edu/dataset/14/breast+cancer dla klasyfikatorów opartych o zbliżony działaniem model lasu losowego