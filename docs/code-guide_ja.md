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
  例えばSevenNetの`omol25_low`/`omol25_high`は精度の高低ではなく、低スピン系と
  高スピン系という**スピン状態での分割**である。
- 重いMLIP packageはbackend内で遅延importし、未使用backendのimport失敗を避ける。

## 分子系のcharge / spin

分子系だけに必要な入力が2つある。系全体の**総電荷**と**スピン多重度**(`2S+1`)である。
ASEにはこれらの標準的な置き場所が無いため`atoms.info`を経由するが、
**backendによって受け取れるかどうかが正反対**なので、ここが最も間違えやすい。

| backend | 分子系の指定 | charge / spin |
|---|---|---|
| `uma` | `task="omol"` | ✅ `atoms.info["charge"]`, `atoms.info["spin"]` |
| `sevennet` | `modal="omol25_low"`など | ❌ sevennetに入力そのものが存在しない |

```python
atoms.info["charge"] = -1   # 総電荷
atoms.info["spin"] = 2      # スピン多重度 2S+1
atoms.calc = get_calculator("uma", task="omol")
```

UMAで特に注意が要るのは、**未設定でもエラーにならない**ことである。fairchemは警告を
logに出したうえで、呼び出し側が渡した`atoms.info`へ`charge=0`/`spin=1`を書き込み、
中性閉殻として計算を続ける。したがってアニオンやラジカルは**無警告で誤った値**が返る。
分子系では毎回明示的に設定する。

SevenNetにはcharge/spinの入力が無いため、イオンや任意の開殻状態は表現できない。
`omol25_high`は高スピン配置で学習されたmodelを選ぶだけで、構造ごとに指定する
多重度ではない。電荷とスピンが計算結果を左右する系ではUMAの`omol`を使う。

なお`get_calculator`にcharge/spin引数は用意していない。SevenNetでは黙って捨てるしか
なくなり、「引数を渡したのに効かない」という最悪の失敗の仕方になるためである。

## backend追加時の順序

1. `backends/mlip/`または`backends/dft/`へ小さなclassを追加する。
2. `registry.py`へ公開名を1行追加する。
3. 対応device、model/task/modal、学習参照レベルをdocstringへ書く。
4. D3を許可する場合は`dispersion.py`と`docs/models.md`を同時に更新する。
   両者のずれは`tests/test_models_doc_sync.py`が検出する。表の1列目には
   policy keyを必ずbacktickで書く。
5. `pyproject.toml`へextraを追加し、`.github/workflows/ci.yml`の
   `extras-resolve`が対象Python全てで解決できることを確認する。
6. modelをdownloadしないmock testと、必要に応じて`slow` single-point testを追加する。

backend間の共通化より、各上流packageへ実際に渡す引数が一目で分かることを優先します。
