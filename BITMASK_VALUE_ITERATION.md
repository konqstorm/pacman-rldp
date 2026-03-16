# Food-bitmask approximate value iteration

Этот вариант реализует **Value Iteration**

- сначала собирается **эмпирическая модель переходов** `N(s,a,s')` на агрегированном состоянии `food_bitmask`;
- затем запускается **discounted Bellman optimality backup**:

`V_{k+1}(s) = max_a sum_{s'} P_hat(s'|s,a) * (R_hat(s,a,s') + gamma * V_k(s'))`

- после этого политика извлекается как **greedy** по полученной функции ценности.

## Две основные функции

В проекте выделены две алгоритм-специфичные функции:

- `train_food_bitmask_value_iteration(...)`
- `eval_food_bitmask_value_iteration(...)`

Они находятся в:

- `src/pacman_rldp/pipelines_food_bitmask_vi.py`

CLI-скрипты `scripts/train_bitmask_vi.py` и `scripts/eval_bitmask_vi.py` — это тонкие обёртки над этими двумя функциями.

## Что сохраняется в `results/important`

После **train** туда сохраняются:
- `train_bitmask_vi_reward_curve.png` — график обучения: score Pacman по train-эпизодам сбора модели;
- `train_bitmask_vi_metrics.json` — train-метрики.

После **eval** туда сохраняются:
- `eval_bitmask_vi_metrics.json` — метрики качества политики на 200 эпизодах;
- `*.gif` — GIF лучшего эпизода, если eval запускался в `render-mode=human`.

## Какие метрики считаются в eval

- `win_rate = wins / 200`
- `mean_return`
- `mean_steps`
- дополнительно: `mean_score`, `unseen_state_fallbacks`, лучший эпизод и массивы `wins/returns/steps/scores`.

## Команды запуска

### Train
```bash
python scripts/train_bitmask_vi.py --config configs/bitmask_value_iteration.yaml
```

### Eval на 200 эпизодах
```bash
python scripts/eval_bitmask_vi.py --config configs/bitmask_value_iteration.yaml --episodes 200 --render-mode none
```

### Eval + GIF лучшего эпизода
```bash
python scripts/eval_bitmask_vi.py --config configs/bitmask_value_iteration.yaml --episodes 200 --render-mode human --gif-title bitmask_best
```
