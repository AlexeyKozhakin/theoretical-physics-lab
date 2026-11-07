# Вычисления и проверки

Здесь находятся проверки вывода и прототип вычисления относительного потенциала. Полная фазовая диаграмма пока не рассчитана.

Проверенное окружение: Python 3.11, NumPy 2.2.5, SciPy 1.17.0, SymPy 1.14.0, mpmath 1.3.0. Корневой `requirements.txt` предназначен для сборки документации; эти научные библиотеки в него не добавлены.

Научные зависимости, включая matplotlib для графиков, зафиксированы отдельно. Установка из корня репозитория в выбранное Python-окружение:

```sh
python3 -m pip install -r cdw-magnetic/code/requirements.txt
```

Запуск из корня репозитория:

```sh
python3 cdw-magnetic/code/check_rabi_reduction.py
python3 cdw-magnetic/code/check_heat_kernel.py
python3 cdw-magnetic/code/check_finite_density.py
python3 cdw-magnetic/code/check_zero_field_limit.py
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/check_relative_potential.py --output cdw-magnetic/code/relative-potential-checks.json
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/check_basis_and_homogeneous.py --output cdw-magnetic/code/basis-and-homogeneous-checks.json
```

Ограничение числа потоков BLAS полезно для многочисленных небольших матриц; на физические формулы оно не влияет. Последняя команда перезаписывает только указанный численный отчёт.

`relative_potential.py` содержит `RelativeBlock` — реализацию формулы из [заметки 11](../notes/11-relative-determinant.md). Пример при запуске из этого каталога:

```python
from relative_potential import RelativeBlock

block = RelativeBlock(field=0.8, mass=0.7, wave=0.4, cutoff=32)
result = block.matsubara(chemical=1.2, temperature=0.2, tolerance=1e-9)
print(result.value, result.frequency_tail_bound)
```

**`frequency_tail_bound` не включает ошибку осцилляторного базиса.** Обязательно повторять расчёт при увеличении `cutoff`; малая частотная ошибка не гарантирует правильный знак полного результата. В сохранённом отчёте есть пример, где знак меняется между 16 и 32 состояниями в каждой башне. УФ-поправка и экстраполяция в проверочном скрипте — диагностика, а не универсальная сертифицированная оценка ошибки базиса.

`basis_bounds.py` добавляет аналитические границы поправки от исключённого базиса. Пример:

```python
from basis_bounds import relative_interval

interval = relative_interval(block, chemical=1.2, temperature=0.2)
print(interval.lower, interval.upper, interval.fixed_background_sign)
```

Верхняя вакуумная оценка использует либо `4*b**2 < M**2 + 2*B`, либо новую оценку исключённого хвоста и связи блоков Шура из заметки 13. Вторая требует `gamma_N > 0` и положительности `L_N - R_N/gamma_N`, поэтому может работать также при сильном смешивании. Если неприменимы обе оценки, верхняя граница бесконечна, нижняя сохраняется. Тепловая оценка требует положительной нижней границы исключённого блока; иначе нужно увеличить базис. `decision_tolerance` ограничивает практическое решение о знаке, но не является интервальным сертификатом округления. Эти результаты сравнивают фиксированный фон, не определяют глобальную фазу.

`homogeneous_potential.py` реализует однородный потенциал (включая отдельную ветку `B=0`), аналитическую оценку теплового хвоста и численный поиск минимума по массе. По умолчанию `g=-pi`, то есть вакуумная масса равна единице. Поиск проверяется уточнением сетки и верхней границы массы; в точках вырождения единственность минимума не предполагается.


`adaptive_search.py` сравнивает точки с уточнённым однородным минимумом, увеличивает базис и сгущает сетку около перспективных точек. Это конечный поиск кандидатов, а не сертификация всех промежутков. При `M=0` волновое число физически несущественно. Отчёт содержит также неопределённые точки и проверки границ области. Проверки включают искусственный минимум между узлами начальной сетки, разбиение большой матрицы на хвост и связь блоков, малую массу и отдельный кандидат на N=64,128,256.

```sh
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/check_adaptive_search.py
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/adaptive_search.py --rounds 3 --max-cutoff 128
MPLCONFIGDIR=/private/tmp/theoretical-physics-lab-mpl OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/plot_adaptive_search.py
```

Команды сохраняют `adaptive-search-checks.json`, `adaptive-search-pilot.json` и график `cdw-magnetic/figures/adaptive-search-pilot.png`. Для графика дополнительно нужен matplotlib. Результат и ограничения: [заметка 13](../notes/13-adaptive-search-and-cdw-candidate.md).


Локальное уточнение одного кандидата и график его окрестности:

```sh
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/refine_cdw_candidate.py
MPLCONFIGDIR=/private/tmp/theoretical-physics-lab-mpl OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/plot_cdw_refinement.py
```

`refine_cdw_candidate.py` использует прямой конечный спектр с поправкой первого вакуумного хвоста как приближённую функцию поиска, затем отдельно проверяет выбранную точку определителем Шура и полными аналитическими границами базиса на N=128,256,512. Встроены проверки сходимости координат и значений, гессиана целевой функции, согласия спектрального и частотного расчётов и отрицательности верхних границ. Это не глобальный поиск. Отчёт: `cdw-candidate-refinement.json`; график: `cdw-magnetic/figures/cdw-candidate-refinement.png`. Подробнее — [заметка 14](../notes/14-local-cdw-minimum.md).


Продолжение кандидатов и независимый бесполевой контроль:

```sh
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/zero_field_potential.py
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/check_zero_field_potential.py
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/continue_cdw_branch.py
MPLCONFIGDIR=/private/tmp/theoretical-physics-lab-mpl OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/plot_cdw_continuation.py
```

`zero_field_potential.py` использует аналитический вакуум, точный поперечный интеграл при T=0 и температурную свёртку с аналитической оценкой хвоста. Скрипт проверок сравнивает её с прямыми двумерными импульсными интегралами, включая независимое УФ-вычитание вакуума. Отчёт: `zero-field-potential-checks.json`.

`continue_cdw_branch.py` ищет локальные кандидаты на базисах 64,128 и проверяет перспективные фоны полными границами на 128,256 (при необходимости 512). Однородный минимум вычисляется заново в каждой точке. При отсутствии кандидата добавляются старты с большей массой; при достижении верхней границы массы окно расширяется. `--recheck-unconfirmed` сохраняет подтверждённые результаты существующего отчёта и перепроверяет остальные. Отчёт `cdw-branch-continuation.json` содержит промежуточные контрольные точки (`completed=false`), фактические окна расширенных поисков, все старты и неопределённые исходы. Ранее сохранённые подтверждённые точки без поля `actual_search_window` используют начальное окно отчёта. Это два сечения локального поиска, не глобальная карта. График: `cdw-magnetic/figures/cdw-branch-continuation.png`. Вывод и ограничения — [заметка 15](../notes/15-continuation-and-zero-field-control.md).


Первая предварительная двумерная карта:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 cdw-magnetic/code/phase_scout.py --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 cdw-magnetic/code/check_phase_scout.py
MPLCONFIGDIR=/private/tmp/theoretical-physics-lab-mpl OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/plot_phase_scout.py
```

`phase_scout.py` использует общий набор 242 спектров на каждое поле для всех 25 химпотенциалов, затем локальные уточнения на N=64,128 и полные границы на 128,256,512 только в перспективных точках. Вакуум однородной массовой сетки также переиспользуется. Параллелизм — независимые CPU-процессы по полям; число потоков BLAS ограничено, чтобы избежать конкуренции за ядра. `--resume` продолжает сохранённый отчёт по завершённым строкам. В `phase-scout.json` поле `completed=false` означает промежуточный файл.

CDW-пометка требует отрицательной полной верхней оценки и минимум двух проверенных базисов. Остальные фазовые пометки обозначают лидеров приближённого поиска с рабочей точностью 1e−7, без глобальной проверки отсутствия других минимумов. Ошибки и неразрешённые верхние границы сохраняются отдельно. `check_phase_scout.py` проверяет сетку и пересечения интервалов, прежние кандидаты, переиспользование спектров и четыре контрольные точки на N=512 с уточнённым однородным поиском; дополнительно перепроверяет неразрешённые точки. Отчёт: `phase-scout-audit.json`. Графики: `phase-scout.png`, `phase-scout-condensates.png` в `cdw-magnetic/figures/`. Метод, числа и пределы интерпретации — [заметка 16](../notes/16-preliminary-phase-map.md).

## Быстрый движок карт

`fast_engine.py` — быстрый поиск минимума $\Omega(M,b)$ для строк карты $(\mu,B)$. Относительный вакуум $\Delta V$ считается один раз на поле и интерполируется сплайном (с кэшем на диске). Нижние уровни берутся из ленточной формы Раби, все $\mu$ и $T$ используют одни и те же уровни. Поиск глобальный на сетке $(M,b)$ 65×69. Кандидаты с выигрышем меньше $-10^{-6}$ перепроверяются без сплайна на двух размерах базиса. Значения не сертифицированы. Описание, проверки и замеры — [заметка 17](../notes/17-fast-engine.md).

```bash
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/fast_benchmark.py
```

`fast_benchmark.py` сравнивает строку $|eB|=0.2$ с `phase-scout.json` и печатает время: без кэша, на 4 процессах, с кэшем, для 121 значения $\mu$ и для $T=0.01,0.005$. Кэш таблиц лежит в `/tmp/cdw-fast-engine-cache`.

## Адаптивная карта

`adaptive_map.py` строит карту $0<\mu\le2$, $0<|eB|\le7$ при $T=0.01$: грубая сетка 60×60 и три уровня деления рёбер с разными метками. Около 4 минут на 4 ядрах. Отчёт — `adaptive-map.json`, графики — `plot_adaptive_map.py` → `cdw-magnetic/figures/adaptive-map.png` (фаза, $M$, $b$; оси $x=|eB|$, $y=\mu$) и `adaptive-map-gain.png`. Подробнее — [заметка 18](../notes/18-adaptive-map.md).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 cdw-magnetic/code/adaptive_map.py --workers 4
MPLCONFIGDIR=/tmp/mpl python3 cdw-magnetic/code/plot_adaptive_map.py
```

## Пакет `magcdw` и однородная диаграмма

Ядро расчётов переносится в пакет `cdw-magnetic/magcdw/` (см. [ROADMAP](../ROADMAP.md)); тесты — `cdw-magnetic/tests/`. Однородная задача — `magcdw.hom` ([заметка 19](../notes/19-homogeneous.md)).

```bash
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q cdw-magnetic/tests
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl python3 cdw-magnetic/code/hom_diagram.py --T 0 --n 240
```
