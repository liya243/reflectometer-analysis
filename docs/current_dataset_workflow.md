# Workflow для текущего датасета

Этот файл фиксирует параметры, которые использовались для текущего локального датасета
`mem1_07_10-11_10_22.dat`. Сам файл данных не коммитится в git.

## Геометрия и временные параметры

- Частота записи рефлектограмм: `10000 Hz`.
- Сервисная зона: примерно `0-110 m`.
- Полезный участок волокна в большинстве расчётов: `110-360 m`.
- Тёмный уровень фотодиода: последние `50 m` каждой трассы.
- Поддержка импульса для деконволюции: `75-85 m`.
- Координата нулевого уровня для формы импульса: `70 m`.
- Длины импульса чередуются, поэтому чётные и нечётные рефлектограммы обрабатываются отдельно.
- Размах пилообразного свипа: `3.125 pm`.
- Принятый период свипа: `76.8 ms`.
- Принятая опорная граница свипа: `0.0919 s`.
- Верхняя граница времени, где ещё учитываются сбросы модуляции: `4.45 s`.
- Последний полный свип, использованный для деконволюции: `4.3159-4.3927 s`.

## Пересоздание цветных карт

```bash
python fiber_colormap_with_resets_even_odd.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --baseline-tail-m 50 \
  --reset-period-override-ms 76.8 \
  --reset-anchor-time-s 0.0919 \
  --shared-reset-detection \
  --reset-detection-end-time-s 4.5 \
  --max-reset-time-s 4.45 \
  --regularize-reset-grid
```

## Пересоздание форм импульсов

```bash
python pulse_profile_even_odd.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --pulse-z-min 75 --pulse-z-max 85 \
  --zero-level-z 70 \
  --baseline-tail-m 50
```

## Восстановление комплексного поля дискретов по последнему свипу

```bash
python solve_complex_amplitudes_from_harmonics.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --pulse-z-min 75 --pulse-z-max 85 --zero-level-z 70 \
  --sweep-span-pm 3.125 \
  --rolling-window 64 --min-period-s 0.03 --max-period-s 0.2 \
  --prominence-sigma 2.0 --refine-window-fraction 0.15 \
  --reset-time-shift-ms 0.0 \
  --reset-period-override-ms 76.8 --reset-anchor-time-s 0.0919 \
  --shared-reset-detection --reset-detection-end-time-s 4.5 --max-reset-time-s 4.45 \
  --regularize-reset-grid --baseline-tail-m 50 \
  --sweep-index -1 --lag-min 2 --lag-max 16 \
  --direct-iters 20 --direct-damping 0.25 --direct-ridge-lambda 1e-3
```

## Сопоставление референса после модуляции с последним свипом

Прямая корреляционная проверка около `6 s` показывает, что лазер остановился примерно около середины
последнего калиброванного свипа: около `1.6 pm` внутри интервала `0..3.125 pm`. Корреляция слабая,
поэтому это sanity-check, а не точное измерение длины волны.

```bash
python match_reference_time_to_last_sweep.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --baseline-tail-m 50 \
  --reset-period-ms 76.8 --reset-anchor-time-s 0.0919 --max-reset-time-s 4.45 \
  --sweep-index -1 --sweep-span-pm 3.125 \
  --reference-time-s 6.0 \
  --reference-half-window-traces 32 \
  --sweep-half-window-traces 4 \
  --exclude-z-min 230 --exclude-z-max 240
```

## Отслеживание фазы пьезоэлемента на двух дискретах

Этот анализ использует комплексное поле, восстановленное по последнему свипу, и выбранную на данный
момент коррекцию дрейфа длины волны. Для текущего датасета лучший найденный кандидат — соседняя пара:

- `230.446111 m`
- `230.965134 m`

```bash
python track_two_discrete_piezo_phase.py mem1_07_10-11_10_22.dat \
  --output-dir analysis_outputs \
  --fiber-z-min 110 --fiber-z-max 360 \
  --baseline-tail-m 50 \
  --candidate-z-min 230 --candidate-z-max 240 \
  --phase-start-time-s 5.5 --baseline-duration-s 0.35 \
  --phase-grid-size 721 --block-size 32 \
  --fit-window-half-width-m 18 --rolling-window 9 \
  --lambda0-nm 1550 --drift-sign 1 \
  --phase-continuity-lambda 0.015 \
  --phase-zero-prior-lambda 0.05 \
  --parity-align-corr-floor 0.35
```
