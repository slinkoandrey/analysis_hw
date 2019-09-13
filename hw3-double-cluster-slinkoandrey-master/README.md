# Домашнее задание № 3
## Восстановление параметров статистического распределения

В этом задании вы будете восстанавливать параметры смешанных статистических распределений, используя два метода: метод максимального правдоподобия и EM-метод. Научным заданием будет выделение двух рассеянных скоплений  [h и χ Персея](https://apod.nasa.gov/apod/ap091204.html) в звёздном поле.

**Дедлайн 12 ноября в 23:55**

Вы должны реализовать следующие алгоритмы в файле `mixfit.py`:

1. **Метод максимального правдоподобия для смеси двух одномерных нормальных распределений.** Напишите функцию `max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3)`, где `x` — массив данных, остальные позиционные аргументы — начальные приближения для искомых параметров распределения, `rtol` — относительная точность поиска параметров, функция должна возвращать кортеж из параметров распределения в том же порядке, что и в сигнатуре функции. Для оптимизации разрешается использовать `scipy.optimize`.

2. **[Expectation-maximization метод](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm)** для той же задачи: `em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3)`.

3. **EM-метод для смеси двух двумерных симметричных нормальных распределений с двумерным равномерным распределением** — τ1 N(µ1, σ1) + τ2 N(µ2, σ2) + (1-τ1-τ2) U, считая параметры равномерного распределения фиксированными. Напишите функцию `em_double_cluster(x, uniform_dens, tau1, mu1, sigma1, tau2, mu2, sigma2, rtol=1e-3)`, где `x` — массив `N x 2`, `uniform_dens` — плотность вероятности равномерного распределения.

Напишите в файле `test.py` по одному тесту для первых двух методов.
Для отладки ваших алгоритмов используйте генераторы случайных чисел: [`scipy.stats.multivariate_normal.rvs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html) и [`scipy.stats.uniform.rvs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html).

Примените последний EM-метод в файле `per.py` для решения задачи нахождения центров и относительного числа звёзд в скоплениях h и χ Персея.
Вам понадобиться модуль [astroquery.vizier](https://astroquery.readthedocs.io/en/latest/vizier/vizier.html) для того, чтобы загрузить координаты звёзд поля.
Координаты центра двойного скопления `02h21m00s +57d07m42s`, используйте поле размером `1.5 x 1.5` градуса.
Пример запроса для получения данных:

```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

center_coord = SkyCoord('02h21m00s +57d07m42s')
vizier = Vizier(
    column_filters={'Bmag': '<13'},  # число больше — звёзд больше
    row_limit=10000
)
stars = vizier.query_region(
    center_coord,
    width=1.5 * u.deg,
    height=1.5 * u.deg,
    catalog='USNO-A2.0',
)[0]
ra = star['RAJ2000']._data  # прямое восхождение, аналог долготы
dec = star['DEJ2000']._data  # склонение, аналог широты
```

В файл `per.json` сохрнаите результаты:

```json
{
  "size_ratio": 1.2,
  "clusters": [
    {
      "center": {"ra": 35.35, "dec": 57.07},
      "sigma": 0.2,
      "tau": 0.25
    },
    {
      "center": {"ra": 36.36, "dec": 57.57},
      "sigma": 0.1,
      "tau": 0.3
    }
  ]
}
```

В файле `per.png` изобразите график рассеяния точек звёздного поля и найденные центры скоплений (они должны быть хорошо видны на фоне звёзд).
