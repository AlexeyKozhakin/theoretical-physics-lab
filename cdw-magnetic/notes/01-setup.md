# 01. Модель, анзац, калибровка

**Проверка и уточнение знаков (2026-11-06):** полный вывод в [07-derivation-operators](07-derivation-operators.md). Ниже приняты знаки линеаризованного лагранжиана $-\sigma-i\gamma^5\pi$, так что гамильтониан имеет массовый член $+\beta M$.

## Лагранжиан

Исходный лагранжиан (1) [CDW], один O(N)-мультиплет 4-компонентных спиноров:

$$
L = \bar q\left[\gamma^\rho i\partial_\rho + \mu\gamma^0\right]q + \frac{G}{N}\left[(\bar qq)^2+(\bar q i\gamma^5 q)^2\right].
$$

Представление $\gamma$-матриц — приводимое 4×4, Appendix A [CDW]: $\gamma^\mu=\mathrm{diag}(\tilde\gamma^\mu,-\tilde\gamma^\mu)$, $\gamma^3,\gamma^5$ из (A6).

Магнитное поле вводим как в [MC], через удлинённую производную:

$$
\partial_\rho \to \nabla_\rho = \partial_\rho + ieA_\rho,\qquad A_{0,1}=0,\ A_2=Bx,
$$

что даёт $B_z=B$, перпендикулярно плоскости. Без ограничения общности $eB\ge0$.

## Анзац и TDP

Анзац (8) [CDW] с $\vec l=(2b,0)$. Используем знак массы из линеаризованного лагранжиана (2) [CDW]; по сравнению с записью (11) знак массового члена выбран противоположным, что соответствует симметрии $M\to-M$. Оператор Дирака:

$$
D = \gamma^0(i\partial_0+\mu) + \gamma^1 i\partial_1 + \gamma^2 i\nabla_2 - M e^{2i\gamma^5 bx}.
$$

## Киральный поворот

Делаем замену (12) [CDW]: $q_w=e^{i\gamma^5bx}q$, $\bar q_w=\bar q\,e^{i\gamma^5bx}$.

- В слагаемом с $\gamma^1\partial_1$ производная действует на $e^{-i\gamma^5bx}$ и даёт $\gamma^1\gamma^5 b=-\gamma^5\gamma^1 b$. Это исправляет знак в первоначальной версии заметки.
- В слагаемом с $\gamma^2$: $\nabla_2=\partial_2+ieBx$ зависит от $x$, но не содержит $\partial_1$, поэтому коммутирует с $e^{\pm i\gamma^5 bx}$. Новых членов не появляется.

Итого в этих соглашениях:

$$
\mathcal D = \gamma^0(i\partial_0+\mu) + \gamma^1 i\partial_1 + \gamma^2 i\nabla_2 - M - \gamma^5\gamma^1 b.
$$

Оператор не зависит от $y$, так что $p_2$ сохраняется. Сдвиг $x\to x-p_2/(eB)$ убирает $p_2$, отсюда стандартное вырождение уровней $eB/2\pi$ на единицу площади.

Вопрос о том, меняет ли поворот меру интегрирования при наличии $A_\mu$, обсуждается в [04-renorm](04-renorm.md).

## Гамильтонова форма

С $\alpha^i=\gamma^0\gamma^i$, $\beta=\gamma^0$, $p_i=-i\partial_i$, $\pi_2=p_2+eBx$:

$$
H = \alpha^1p_1 + \alpha^2\pi_2 + \beta M + b\,\Gamma,\qquad \Gamma=\gamma^0\gamma^5\gamma^1.
$$

Свойства $\Gamma$ (проверено по (A5)–(A6)): $\Gamma^2=1$, $[\Gamma,\alpha^1]=[\Gamma,\beta]=0$, $\{\Gamma,\alpha^2\}=0$.
