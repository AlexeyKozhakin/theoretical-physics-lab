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
