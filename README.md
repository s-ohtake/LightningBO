# LightningBO

LightningBOは、ベイズ最適化（Bayesian Optimization）を簡単に実装・拡張できるPythonライブラリです。
本ライブラリは、ブラックボックス関数の最適化やハイパーパラメータ探索など、試行回数を抑えつつ効率的に最適解を見つけたい場面で活用できます。

## 特徴

- JAXベースの高速な計算
- 連続・離散パラメータ両対応
- 期待値改善（EI）などの獲得関数
- 拡張性の高い設計

## 使い方例

```python
import LightningBO.bayesian_optimization as bo
import LightningBO.bayesian_core as bc

# 最適化したい関数
def f(x):
	return x**2 - 4*x + 5

# 探索空間の定義
domain = {"x": bc.Continuous(-10, 10)}

# Optimizerの作成
optimizer = bo.Optimizer(domain=domain, maximize=False)

# 初期点
params = {"x": [0.0, 2.0, 5.0]}
ys = [f(x) for x in params["x"]]
state = optimizer.init(ys, params)

# 最適化ループ
for step in range(20):
	new_params = optimizer.sample(None, state)
	y = f(**new_params)
	state = optimizer.fit(state, y, new_params)

print("最適値:", state.best_score)
print("最適パラメータ:", state.best_params)
```

## 主な用途

- 機械学習モデルのハイパーパラメータ最適化
- 実験・シミュレーションのパラメータ探索
- その他ブラックボックス関数の最適化
