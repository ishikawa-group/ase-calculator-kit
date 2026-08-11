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
- `backends/mlip/`: MLIPごとのmodel/task/modal/headと上流APIの違いを明示する。
- `backends/dft/`: YAML条件をASEのVASP/QE Calculatorへ変換する。
- `device.py`: `auto`をcpu/cuda/mpsのいずれかへ解決する。
- `dispersion.py`: 学習参照汎関数とD3二重計上防止の唯一の実装。
- `docs/models.md`: `dispersion.py`の化学的根拠を人が確認する表。

## 化学的に重要な約束

- ASEの標準単位をそのまま使う。energyはeV、forceはeV/Å、stressはeV/Å³。
- D3のパラメータは、モデルの学習参照汎関数に合わせる。
- 学習データがすでに分散相互作用を含むモデルへD3を重ねない。
- task/modal/headは単なる速度設定ではなく、対象化学系とDFT参照レベルの選択である。
  例えばSevenNetの`omol25_low`/`omol25_high`は精度の高低ではなく、低スピン系と
  高スピン系という**スピン状態での分割**である。MACE-MH-1のheadも同様に、
  PBE / r2SCAN / ωB97Mという参照レベルそのものを選んでいる。
- **上流が黙って無視する引数を、D3の汎関数選択に使ってはいけない。** sevennは
  単一fidelityのmodelに`modal`を渡しても警告だけ出して捨てるため、0.5.2以前は
  `model="7net-0", modal="matpes_r2scan"`がPBEのmodelにr2SCANのD3を足していた。
  MACEのheadも同じ形の罠である。現在はどちらも`"auto"`で解決し、
  **解決後の値**をpolicy keyに使い、その checkpoint が使えない選択肢を明示的に
  渡した場合はエラーにしている。単一head/単一fidelityのmodel
  (`medium-omat-0`, `7net-omat`など)はmodel名そのものがkeyになる。
- 逆にSevenNetの`7net-omni` / `-i8` / `-i12`は同じ学習レシピの容量違いであり、
  参照レベルは変わらない(`modal`の表はそのまま使える)。ただし別modelなので、
  1つの計算キャンペーン内で混ぜるとエネルギーの比較ができない。
- 重いMLIP packageはbackend内で遅延importし、未使用backendのimport失敗を避ける。

## MACEは専用の仮想環境が必要

MACEは0.5.0から利用できるが、**他のMLIP backendと同じ環境には入らない**。
`mace-torch`は`e3nn==0.4.4`を固定し、`sevenn`・`fairchem-core`・`mattersim`は
`e3nn>=0.5.0`、`nequip`は`>=0.6.0`を要求するため、両立する解が存在しない。
したがって`mace` extraは`all`に含めず、専用のvenvへ入れる。

```bash
python -m venv .venv-mace
.venv-mace/bin/pip install "ase-calculator-kit[mace]"
```

この環境でもfactoryやD3方針の扱いは全く同じで、単に他のMLIPが
`MissingDependencyError`になるだけである。逆に通常の環境で`mace`を要求した場合も、
同じ例外がこの制約を説明する。

### headの検証をこちら側で行う理由

MACE-MH-1は1つのcheckpointに6つのheadを持ち、**headの選択がそのままDFT参照レベルの
選択**になる(`omat_pbe`=PBE(+U)、`matpes_r2scan`=r2SCAN、`omol`=ωB97M-VV10など)。
ここで問題なのは、上流の`MACECalculator`が**未知のheadを渡してもエラーにしない**点である。
警告をlogに出したうえでcheckpointの最後のheadへ勝手に切り替えて計算を続けるため、
綴り間違いは「もっともらしいが別汎関数の値」として返る。UMAのcharge/spin未設定と同じ
失敗の形なので、`backends/mlip/mace.py`がdownloadの前にheadを検証して`ValueError`にする。

なおhead名は配布されている`mace-mh-1.model`から読み出した実際の一覧であり、
model cardが載せている`rgd1_b3lyp`はこのcheckpointには存在しない。

### GPUアクセラレータを`auto`が"測ってから"決める理由

`accelerator="auto"`(既定)はCUDA上でcuequivarianceを使うが、
**「importできるか」では判断しない**。実測で2回とも裏切られたためである。

- Tesla V100(sm_70)ではimportもcalculatorの生成も成功し、**最初のenergy評価**で
  `cudaErrorNoKernelImageForDevice`で落ちる。配布kernelが新しいarchのみのため。
  import判定だと「走り始めてから壊れるcalculator」を返してしまう。
- ACEsuit/mace#1298では、multi-head checkpointでcuequivarianceが
  **例外を出さずに** -200 eVのところを+5500 eVで返す報告がある。

そこでautoは両方のmodelを作り、2原子のcellで energy を突き合わせ、一致した場合だけ
加速版を採用する。追加コストはmodel生成1回と2原子の1点計算2回で、
cuequivarianceが入っていない環境では probe 自体を行わない。
V100で実測したところ、autoは警告を出してfallbackし、`bulk("Cu")`で
CPU float64と1e-13 eV以内で一致した。失敗後もCUDA contextは壊れず、
fallbackしたcalculatorはそのまま正しい値を返すことも確認している。

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
   他のbackendと依存が衝突する場合はMACEと同様に`all`へ入れず、
   `tests/test_packaging.py`で「`all`に含まれないこと」を固定する。
6. modelをdownloadしないmock testと、必要に応じて`slow` single-point testを追加する。

backend間の共通化より、各上流packageへ実際に渡す引数が一目で分かることを優先します。
