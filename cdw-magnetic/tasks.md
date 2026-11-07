# Постановки задач

Обе постановки приведены дословно, на языке оригинала. Изменено только одно: локальные пути к файлам заменены названиями статей. [CDW] = `3dimCDW+L_v4-4.pdf`, [MC] = arXiv:1307.7265.

Модель и внешнее поле в обеих ветках одинаковые, различается метод:

- **Ветка A — TDP с CDW-анзацем.** Подставить анзац (8) в поле и вычислить TDP. Заметки: [notes/](notes/01-setup.md).
- **Ветка B — анализ устойчивости.** Проверить устойчивость однородного основного состояния в поле относительно малых неоднородных флуктуаций, как в Appendix F [CDW]. Заметки: [notes/stability/](notes/stability/00-plan.md).

---

## Ветка A: TDP с CDW-анзацем (получена 2026-11-04)

> Take Lagrangian (1) from the file and consider the case without compactification, in (2+1) dim and with two infinite dimensions. And then consider the magnetic field perpendicular to the plane of (x,y), where x, y are spatial dim in (2+1), in the same way it is done in the paper [MC]. So take Lagrangian (1) with mu and assume the ansatz (8) from the file with l = (2b, 0) as in file, but add the perpendicular magnetic field as in [MC] by adding to derivative \partial a vector potential A_mu: A_{0,1} = 0, A_2 = B x, A_3 = 0$, i.e. the spatial components $B_{x,y,z}$ of an external magnetic field have the form $B_x = 0, B_y = 0, B_z = B$, B=const. Try to calculate the TDP of the model in this case.

Уточнения автора: [questions.md](questions.md), R2–R5.

## Ветка B: анализ устойчивости (получена 2026-11-05)

> Take Lagrangian (1) from the file 3dimCDW... and consider the case without compactification, in (2+1) dim and with two infinite dimensions. And then consider the magnetic field perpendicular to the plane of (x,y), where x, y are spatial dim in (2+1), in the same way it is done in the paper https://arxiv.org/abs/1307.7265. So take Lagrangian (1) with mu and assume the ansatz (8) from the file 3dimCDW... with l = (2b, 0) as in file, but add the perpendicular magnetic field as in https://arxiv.org/abs/1307.7265 by adding to derivative \partial a vector potential A_mu: A_{0,1} = 0, A_2 = B x, A_3 = 0$, i.e. the spatial components $B_{x,y,z}$ of an external magnetic field have the form $B_x = 0, B_y = 0, B_z = B$, B=const. Do not consider two O(N) multiplets as in https://arxiv.org/abs/1307.7265 that was done to include the spin in the system.
>
> Perform stability analysis as it is done in Apendix F of the file 3dimCDW..., i. e. find similar formula for S^(2)(ε, ω) but in the case of homogeneous perpendicular magnetic filed as I described above

## Ветка B: уточнённая постановка (получена 2026-11-05)

Заменяет предыдущую формулировку ветки B по содержанию; прежний текст сохранён выше. Текст приведён дословно в исходном LaTeX (в блоке кода, чтобы ничего не исказилось при отображении). «2dimCDW» в тексте означает [CDW] = `3dimCDW+L_v4-4.pdf`; https://arxiv.org/abs/1307.7265 — это [MC].

Разбор постановки и вопросы — [notes/stability/01-setup.md](notes/stability/01-setup.md).

```latex
We consider a (2+1)-dimensional NJL-type model wih the following Lagrangian density,
\begin{equation}
L=\bar q\Big [\gamma^\rho\mathrm{i}D_\rho
+\mu\gamma^0\Big ]q+ \frac {G}{N}\Big
[(\bar qq)^2+(\bar q\mathrm{i}\gamma^5 q)^2 \Big ],
\label{1}
\end{equation}
where the fermion field $q(t,\vec r)$ (here $t\equiv x_0$ is a time-coordinate and $\vec r\equiv (x,y)$ is a radius-vector  of a point from the two-dimensional spatial plane)
as well as a reducible four-dimensional spinor multiplet of the (2+1)-dimensional Lorentz group, i.e. $q(t,\vec r)\equiv q_{si}(t,\vec r)$, where $s$ and $i$ are the spinor ($s=1,2,3,4$) and $U(N)$-group indices ($i=1,...,N$), correspondingly. Moreover, there the
summation over $\rho =0,1,2$ is implied, $\mu$ is the fermion number chemical potential. Expressions
as well as the algebra of the
$\gamma^\rho$ matrices ($\rho=0,1,2,5$) are the following.

{\bf $\gamma$ matrices.} The two-dimensional irreducible representation of the 3-dimensional
Lorentz group SO(2,1) is realized by the following $2\times 2$
$\tilde\gamma$-
matrices :
$$
\tilde\gamma^0=\sigma_3=
\left (\begin{array}{cc}
1 & 0\\
0 &-1
\end{array}\right ),\,\,
\tilde\gamma^1=i\sigma_1=
\left (\begin{array}{cc}
0 & i\\
i &0
\end{array}\right ),\,\,
\tilde\gamma^2=i\sigma_2=
\left (\begin{array}{cc}
0 & 1\\
-1 &0
\end{array}\right ),
\eqno (A.1)
$$
acting on two-component Dirac
spinors.

In our case the Dirac spinors $q$ have the following
form:
$$
q(x)=
\left (\begin{array}{cc}
\psi_{1}(x)\\
\psi_{2}(x)
\end{array}\right ),
\eqno (A.4)
$$
with $\psi_1,\psi_2$ being two-component spinors.
In the reducible four-dimensional spinor representation one deals
with
(4$\times$4) $\gamma$-matrices:
$\gamma^\mu=diag(\tilde\gamma^\mu,-\tilde\gamma^\mu)$, where
$\tilde\gamma^\mu$

where as well
$$
D_{\mu}=\partial_{\mu}+ieA_{\mu}
$$
 where $A_{\mu}=(A_0,A_1,A_2)$, in our case $A_0=A_1=0$ and $A_2=Bx_1$ where $x_1$ is the first coordinate from plane ($x_1$, $x_2$).


Now you can see that in Appendix F "STABILITY OF HOMOGENEOUS VACUUM IN 3D NJL-TYPE MODEL UNDER SMALL INHOMOGENEOUS FLUCTUATIONS" of the file 2dimCDW there has been done stability analysis of (2+1) NJL model.

Obtain the formula for $S^2$ in the similar way as it is done in Appendix F but in the case of the set up I discussed above with Lagrangian (1). The difference with Appendix F is that there is now homogeneous magnetic field B. The phase structure in the case of non-zero B for the homogeneous condensate $\sigma$ is obtained in the framework GN model in  https://arxiv.org/abs/1307.7265. The  phase structure of NJL model in the case of non-zero B is the same as of GN model in this case since there is no condensation of $\pi$ field. For $S^2$ we would like to have maximally simplified expression, maybe expresssed as integral or in some special functions. Then we want to explore the sign of $S^2$ in order to find if there is instability of homogeneous ground state with respect to inhomogeneous fluctuations.
```

## Ветка B: окончательная постановка (получена 2026-11-07)

**Заменяет обе предыдущие формулировки ветки B по содержанию** (они сохранены выше). Полный текст — дословно, в исходном LaTeX: [stability-task-v3.tex](https://github.com/AlexeyKozhakin/theoretical-physics-lab/blob/main/cdw-magnetic/stability-task-v3.tex).

Документ самодостаточен:
- модель (1) с $D_\mu=\partial_\mu+ieA_\mu$, $A_0=A_1=0$, $A_2=Bx_1$, $B=\text{const}$ (знак $B$ или $eB$ можно фиксировать);
- явные γ-матрицы (A.1), (A.4) и $\gamma^5$ (A6);
- раздел о модели и однородном TDP при $B=0$ с перенормировкой (16)–(17), (21);
- полный вывод устойчивости при $B=0$ — аналог Appendix F [CDW], формулы (F1)–(F33).

**Что требуется** (раздел Task):
1. Найти пропагатор в постоянном поле для лагранжиана (1).
2. Повторить выкладку $B=0$ с этим пропагатором: шпуры, упрощение, по возможности взять интегралы и суммы.
3. Посчитать $S^{(2)}$ во всех случаях: $M$ генерируется и не генерируется, при всех $\mu$.
4. Член с $G$: при $M\neq0$ исключить уравнением щели, при $M=0$ — перенормировать, чтобы осталась только $g$.
5. Диагонализовать матрицу 2×2 по $(\sigma,\pi)$, как в (F29)–(F32).
6. Получить максимально упрощённое выражение (интегралы, спецфункции) и исследовать знак $S^{(2)}$ — есть ли неустойчивость однородного основного состояния.

Замечания к тексту и принятые решения — [questions.md](questions.md), R13–R14. Разбор и план — [notes/stability/00-plan.md](notes/stability/00-plan.md).
