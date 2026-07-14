# コードを読むためのガイド

このパッケージは「ASEの構造に、指定した計算器を取り付ける」処理だけを担当します。
エネルギーや力を独自に再計算・変換する層ではありません。返り値は常に標準の
`ase.calculators.calculator.Calculator`です。

## 入口から計算まで

```text
get_calculator("sevennet", modal="matpes_pbe")
  └─ factory.py       名前を検証してbackendを選ぶ
      └─ registry.py  公開名とbackend classの対応表
          └─ backends/mlip/sevennet.py
              ├─ 学習汎関数とD3方針を先に検証
              ├─ cpu/cuda/mpsを決定
              ├─ SevenNet本体を遅延importして生成
              └─ 必要な場合だけD3 Calculatorと加算
```

VASPとQuantum ESPRESSOも同じ入口を使いますが、計算条件はPythonの任意キーワードではなく
YAMLで明示します。最終的に使われた条件は`write_resolved_config=True`で保存できます。

## 各モジュールの責務

- `factory.py`: 公開API。calculator名の検証とbackend呼び出しだけを行う。
- `registry.py`: 名前とbackendの対応。新しいbackendを追加する際の索引。
- `backends/mlip/`: MLIPごとのmodel/task/modalと上流APIの違いを明示する。
- `backends/dft/`: YAML条件をASEのVASP/QE Calculatorへ変換する。
- `device.py`: `auto`をcpu/cuda/mpsのいずれかへ解決する。
- `dispersion.py`: 学習参照汎関数とD3二重計上防止の唯一の実装。
- `docs/models.md`: `dispersion.py`の化学的根拠を人が確認する表。

## 化学的に重要な約束

- ASEの標準単位をそのまま使う。energyはeV、forceはeV/Å、stressはeV/Å³。
- D3のパラメータは、モデルの学習参照汎関数に合わせる。
- 学習データがすでに分散相互作用を含むモデルへD3を重ねない。
- task/modalは単なる速度設定ではなく、対象化学系とDFT参照レベルの選択である。
- 重いMLIP packageはbackend内で遅延importし、未使用backendのimport失敗を避ける。

## backend追加時の順序

1. `backends/mlip/`または`backends/dft/`へ小さなclassを追加する。
2. `registry.py`へ公開名を1行追加する。
3. 対応device、model/task/modal、学習参照レベルをdocstringへ書く。
4. D3を許可する場合は`dispersion.py`と`docs/models.md`を同時に更新する。
5. modelをdownloadしないmock testと、必要に応じて`slow` single-point testを追加する。

backend間の共通化より、各上流packageへ実際に渡す引数が一目で分かることを優先します。
