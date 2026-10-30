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

VASP・Quantum ESPRESSO・CP2Kも同じ入口を使いますが、計算条件はPythonの任意キーワードではなく
YAMLで明示します。最終的に使われた条件は`write_resolved_config=True`で保存できます。

## 各モジュールの責務

- `factory.py`: 公開API。calculator名の検証とbackend呼び出しだけを行う。
- `registry.py`: 名前とbackendの対応。新しいbackendを追加する際の索引。
- `backends/mlip/`: MLIPごとのmodel/task/modal/headと上流APIの違いを明示する。
- `backends/dft/`: YAML条件をASEのVASP/QE/CP2K Calculatorへ変換する。
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

## MatGLのMatPESモデル

`backends/mlip/matgl.py`がTensorNetの2つのPyGモデルを共通処理する。
短縮名`matpes-pbe` / `matpes-r2scan`を正式な日付入りモデル名へ解決してから、
同じ解決結果でD3の汎関数を選ぶ。既存の`chgnet.py`は従来のCHGNetを使う。
MatGL版CHGNetは`matgl_chgnet.py`の`matgl-chgnet`で暫定対応する。
MatGL 4.0.3の既知の`forward`ソースをSHA-256で確認し、三体幾何の
`no_grad()`ブロックだけをASTで除去して、読み込んだモデルのインスタンスに束縛する。
クラスやモジュール全体を変更せず、重みは検証済みHF revisionを既定値とする。
これは薄いファクトリという原則の限定例外で、再学習済みモデルではない。
上流の再学習モデル公開後に検証し、将来のkitリリースで置き換える。
M3GNetは引き続き公開対象に含めない。

MatGLの`PESCalculator`は応力の既定単位がGPaなので、`stress_unit="eV/A3"`と
`use_voigt=True`を指定する。これによりD3と加算したときもstressのshapeが`(6,)`で揃う。
MPSはfloat64を保持できないため、モデルを`.float()`にしてから`.to("mps")`する。
CPUでのdtypeやPyTorchのglobal defaultは変更しない。

両バックエンドの入力変換は`_matgl_pbc.py`で共通化する。MatGL 4.0.3は通常の
変換経路で部分PBCを全非周期として扱うため、部分PBC・非周期系にはASEの近傍リストを
使い、周期像のシフトと実際のセルを渡す。全周期系は上流の変換をそのまま利用する。
入力の`Atoms`やPBCは変更しない。非周期方向の欠けたセルベクトルだけを内部で補完し、
D3併用時もE/Fを取得できるようにする。体積が定義されない系では応力を返さず、
`get_stress()`はASEの`PropertyNotImplementedError`となる。真空付きセルの応力は
セル全体の体積で規格化する。D3側が応力を提供しない全非周期系もE/Fのみとなる。
この入力アダプタはMatGL経由のモデルだけに適用し、共有D3の数値設定は変更しない。

検証は`temp/`で行い、同一のCu・Si・NaCl構造について単位、shape、有限差分、
上流出力との一致、セル複製時のenergyの示量性、CPU/MPSの一致、D3加算を確認する。
検証結果は[`matgl-validation.md`](matgl-validation.md)に記録する。

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

## OrbMolは`all`に入れないが、環境は分けない

`orb`もMACEと同じく`all`から外してあるが、理由は依存衝突ではない。orb-modelsは
他のbackendと問題なく同居する。外す理由は**Pythonのバージョン**である。
orb-modelsが`dm-tree==0.1.8`を厳密固定しており、dm-tree 0.1.8のwheelは全
プラットフォームでcp312までしか存在しない。3.13/3.14ではsource buildに落ちて
失敗するため、`all`に入れると`all`自体が3つのうち2つのPythonでinstallできなく
なってしまう。

modelそのものは3.13で問題なく動く。orb-modelsがdm-treeから使っているのは
`tree.flatten`と`tree.map_structure`だけで、dm-tree 0.1.10にはcp313/cp314の
wheelがある。実測でも3.13 + dm-tree 0.1.10は3.12の参照energyを最後の桁まで
再現した。pipには依存を上書きする機構が無く、PyPIはgit直接参照を禁じているので、
この事実はpackageのmetadataには書けない。READMEに`uv pip install --override`の
手順を置いてあるのはそのためで、forkはしない。

```bash
uv venv --python 3.12 .venv-orb
uv pip install --python .venv-orb/bin/python "ase-calculator-kit[orb]"
```

`uv pip compile`はsource buildを計画するだけで成功してしまうので、
`extras-resolve`には`--only-binary dm-tree`でwheelの有無を見る段を足してある。
上流が固定を緩めた時点でここが落ち、READMEの表を直す合図になる。

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
| `orb` | `model="orbmol-v2"` | ✅ `atoms.info["charge"]`, `atoms.info["spin"]`（**必須**） |
| `sevennet` | `modal="omol25_low"`など | ❌ sevennetに入力そのものが存在しない |

```python
atoms.info["charge"] = -1   # 総電荷
atoms.info["spin"] = 2      # スピン多重度 2S+1
atoms.calc = get_calculator("uma", task="omol")
```

kitで生成したUMA/eSEN omol、分子用MACE、OrbMolは両キーを必須にする。
`atoms.info["spin"]`は多重度で、整数・電子数との整合性も検証する。
上流単独の中性一重項への補完は利用しない。

0.6.0の共通ラッパーは計算時の電子状態と実効電場を独立して保存する。
OrbMol 0.7.0が見落とすcharge/spin変更と、MACEの配列形式の電場変更を検出し、
キャッシュ使用前に再計算要否を判定する。元のAtomsは変更しない。

SevenNetにはcharge/spinの入力が無いため、イオンや任意の開殻状態は表現できない。
`omol25_high`は高スピン配置で学習されたmodelを選ぶだけで、構造ごとに指定する
多重度ではない。電荷とスピンが計算結果を左右する系ではUMAの`omol`を使う。

なお`get_calculator`にcharge/spin引数は用意していない。SevenNetでは黙って捨てるしか
なくなり、「引数を渡したのに効かない」という最悪の失敗の仕方になるためである。

## UMAのfine-tune（r²SCAN・ωB97M-D4rev）

fairchem-core 2.24で`uma-s-1p2p1`の単一task fine-tuneが2つ入った。どちらも
UMA Nitroで動き、`model=`で選ぶ。既定の`task="auto"`がそのmodelの唯一のtaskを選び、
別のtaskを渡すとdownload前に`ValueError`にする。

| `model` | `task` | 理論レベル | 用途 |
|---|---|---|---|
| `uma-s-1p2p1-ft-r2scan` | `omat` | r²SCAN、分散なし、**スピン分極なし** | 無機・混合系の大半 |
| `uma-s-1p2p1-ft-d4rev` | `omol` | ωB97M-D4rev、分散込み | 分子、非共有結合、分子性液体・結晶 |

**D3の方針はtaskではなくmodelで決まる。** `dispersion.py`の表は普段
`("uma", task)`で引くが、fine-tuneは同じtask名で理論レベルが変わる。`omat`で引くと
r²SCANのmodelにPBEのD3を足してしまうので、fine-tuneだけはmodel名をkeyにする
（`_resolve_task`）。`ft-r2scan`は`xc=r2scan`のD3(BJ)を許可し（fairchemの推奨）、
`ft-d4rev`はD4revが参照データに入っているので拒否する。

`ft-r2scan`は全系を非磁性として扱うので、磁性系や凝集エネルギー（孤立原子は磁性を
持つことが多い）には使わず、`uma-s-1p2p1`の`omat`を使う。`ft-d4rev`は`omol`と
同じく電荷と多重度が必須である。どちらも`uma-s-1p2p1`より少ないデータで学習されており、
孤立原子の参照エネルギーも同梱されていない。

## UMAのinference_settings

`inference_settings=`はfairchemの`get_predict_unit()`の引数で、UMAの速度と精度の
交換条件がここにある。calculatorを作る時にしか指定できない。

| 値 | CUDAでの精度 | model | 用途 |
|---|:--:|---|---|
| `"nitro"`（既定） | TF32 | UMA Nitro（専用kernel + CUDA graph） | MD・構造最適化（系が固定） |
| `"default"` | TF32 | fairchem側の`"nitro"`の別名 | — |
| `"multigpu"` | TF32 | merge_mole + compile | 複数worker（`workers > 1`） |
| `"high-precision"` | FP32 | merge_mole + compile | model比較、Hessian |
| `"batch"` | FP32 | mergeしない | 構造を次々変える場合 |
| `"traineval"` | FP32 | mergeしない、graph生成v1 | fairchemの学習・評価値の再現 |

**0.6.3から既定はUMA Nitroである**（fairchem-core 2.24が必要）。Nitroは
`uma-s-1p2p1`専用の推論modelで、compute capability 8.x以上のCUDA GPUで1系ずつ
計算する。使えない要求（別checkpoint、CPU、古いGPU、composition・task・電荷・
スピンの変更）ではfairchemが`UMA Nitro cannot serve this request`と警告して
`"multigpu"`に落ち、必要ならさらにmergeしないmodelに落ちる。計算自体は成功する。
CPUでは常にこの経路になる。`"turbo"`はfairchem-core 2.24で削除されたので、kitも
`"nitro"`/`"multigpu"`を案内する`ValueError`で拒否する。

fairchem-core 2.24.0・torch 2.13.0、H100（MIG 3g.47gb）で
`tests/tsubame_validation_063_nitro.py`により実測した。fcc Cuをrattleした同一系
12フレーム、`task="omat"`、定常は最後の8フレーム平均、差は`"high-precision"`（FP32）比:

| 設定 | 27原子: 初回 | 定常 | 512原子: 定常 | 最大⊿E | ⊿F rmse |
|---|--:|--:|--:|--:|--:|
| `"high-precision"` | 37.5 s | 21.8 ms | 96.6 ms | — | — |
| `"nitro"`（既定） | 9.3 s | **5.4 ms** | **31.4 ms** | 0.15 meV/atom | 0.0004 eV/Å |
| `"multigpu"` | 25.4 s | 13.3 ms | 41.7 ms | 0.14 meV/atom | 0.0004 eV/Å |
| `"batch"` | 0.26 s | 55.5 ms | 269 ms | 0.0003 meV/atom | 0.0000005 eV/Å |

初回の時間は前回の実行でTriton/inductorのcacheが埋まった状態の値で、cacheが空の
状態では`high-precision`85.5 s、`nitro`21.3 s、`multigpu`60.6 sだった。定常の
`multigpu`は別GPUでの実行間で13.3 msと26.9 msとばらついたが、NitroとFP32の比は
安定していた（今回5.4対21.8 ms、1回目4.1対22.8 ms）。

NitroはFP32経路より27原子で4倍、512原子で3倍速く、誤差は`multigpu`と同じTF32由来の
大きさである。既定で実際に`UMANitroModel`が動いたこと、空孔（元素比が同じ）では
Nitroが続くこと、Cu1個をNiに置換すると`Compositions differ from merged model`で
fallbackしつつ計算が返ることも確認した。

**NitroはTF32で計算する。つまり精度を落としている。** 0.6.2まではTF32をopt-inに
していた。このpackageはmodel比較のためにあり、0.1 meV/atom程度の数値誤差はmodel差と
同じ桁だからである。Nitroでは利得が数%ではなくなったので、既定はfairchemに合わせた。
fairchemは該当するとき`Running in a reduced-precision mode`とlogに出す。
**model同士を比較する数値には`inference_settings="high-precision"`を使う。**
FP32のfast pathである。Nitroは二階微分を持たないので、Hessianにもこちらを使う。
TF32はUMAのforward計算を囲むcontext managerの中だけで有効になり、抜けるときに
`torch.get_float32_matmul_precision()`を戻す。同じprocessの他のcalculatorには
漏れない。CPUでは何も起きない。

**fast pathは系が変わらない前提である。** composition・task・電荷・スピンが
固定でなければならない。MDは満たすが、構造をループする用途は満たさない。fairchemは
気づいてfallbackし警告を出す。エラーではないが、Nitroの構築やmergeとcompileを
払って捨てているだけなので、そこは`"batch"`が正しい。fairchem-core 2.22のCPUでは
1点計算あたり18.8秒対2.6秒だった。

名前は`backends/mlip/fairchem.py`側で検証する。上流は`assert`で見ているため
`python -O`で消えるからである。名前で足りない設定（`max_atoms`など）は
`InferenceSettings`オブジェクトをそのまま渡せる。

## backend追加時の順序

1. `backends/mlip/`または`backends/dft/`へ小さなclassを追加する。
2. `registry.py`へ公開名を1行追加する。
3. 対応device、model/task/modal、学習参照レベルをdocstringへ書く。
4. D3を許可する場合は`dispersion.py`と`docs/models.md`を同時に更新する。
   両者のずれは`tests/test_models_doc_sync.py`が検出する。表の1列目には
   policy keyを必ずbacktickで書く。
5. `pyproject.toml`へextraを追加し、`.github/workflows/ci.yml`の
   `extras-resolve`が対象Python全てで解決できることを確認する。
   他のbackendと依存が衝突する場合はMACEと同様に、対象Pythonのwheelが無い場合は
   orbと同様に`all`へ入れず、`tests/test_packaging.py`で「`all`に含まれないこと」
   を固定する。解決できてもwheelが無いことはあるので、疑わしい依存は
   `--only-binary <名前>`で実際のinstall結果を確認する。
6. modelをdownloadしないmock testと、必要に応じて`slow` single-point testを追加する。

backend間の共通化より、各上流packageへ実際に渡す引数が一目で分かることを優先します。


## 分子PySCFバックエンド（0.5.9）

`pyscf` / `gpu4pyscf` はVASP・QEと同じconfig-only factoryに登録する。
`backends/dft/pyscf.py` はASEの構造・単位を上流APIへ接続する薄いアダプターで、
量子化学の計算式は実装しない。非周期系限定、電荷とspinまたは多重度をYAML／atoms.infoから受け取り、
未収束では `CalculationFailed` を返す。GPU要求をCPUに黙って変更しない。

エネルギーはEhからeV、力は勾配の符号を反転しEh/BohrからeV/Åへ変換する。
構造が変わればASEのキャッシュを無効化する。forceを先に要求すると同じSCFの
energyを再利用できる。VV10とD3/D4の重複は拒否し、分散・溶媒・ECPの計算は
PySCF/GPU4PySCFの実装に委譲する。設定項目と実測結果は [pyscf.md](pyscf.md)。
PySCFの追加テストは境界の4件に限定し、実計算検証を通常CIへ大量追加しない。


`atoms.info["spin"]` はUMA／OrbMol同様に多重度であり、PySCF spinへは1を引く。
YAMLと異なるときはキャッシュを返す前にもエラーにし、infoだけの変更でも再計算する。
解決済みYAMLは計算時の電子状態まで記録し、metadataは取得元も保持する。
`esen` はfairchemを共有する独立バックエンドで、OMol25の3 checkpointのみを許可する。
既定はsmall conservingで、directモデルの力はエネルギー勾配とは限らない。

## 0.6.0のPySCF実行状態

同一構造のenergy→forcesは同じ収束済みSCFを使う。`retain_scf: true`では
そのSCFをHessianにも利用し、`reset()`で解放する。既定では力の取得後に解放し、
後からHessianを要求した場合はSCFを作り直す。
`reuse_density`は明示的に有効化した場合だけ、互換な過去の軌道を投影して初期密度にする。
チェックポイントの物理条件・基底の内容・電子状態は計算時のスナップショットと照合する。
未収束の再開は明示的に許可する必要があり、失敗診断と有効な計算結果を混同しない。
設定検証と保存・診断の補助処理は`_pyscf_support.py`、実行順序は`pyscf.py`に置く。
詳しい契約と検証範囲は[pyscf.md](pyscf.md)と[validation-records.md](validation-records.md)。


## CP2K backend

`backends/dft/cp2k.py` の `Cp2kBackend` はASE CP2Kの検証用サブクラスを返す。
DFTのconfig-only規則は同じで、`profile.command`・`profile.data_dir`を必須とし、
環境変数へのフォールバックを使わない。`config.py`はCP2KをQEと別に検証する。
`xc`・`basis_set`・`pseudo_potential`・`cutoff`のキーを必須とし、`null`は雛形に
委ねる。cutoffはeV。`pseudo_potential: auto`は関数に対応する擬ポテンシャルが
見つからない時のGTH-PBEへの置換を避けるため拒否する。
parametersはASE `CP2K.default_parameters`のキーだけ許す。データファイルは
`data_dir`基準の絶対パスへ解決し、存在を確認する。`directory`を絶対パスにし、
labelは`cp2k`として渡す。ASEが解決したlabelを`.inp`・`.out`・PROJECTに使うため、
出力接頭辞が作業ディレクトリを指す。

電子状態は方式(b)のYAML限定。`charge`は整数、`multiplicity`は1以上の整数で、
多重度>1なら未指定/nullの`uks`をTrueにし、明示Falseはエラーにする。
`atoms.info`のcharge/spinは整合確認専用で、spinは多重度。存在する場合は
対応する明示的な非null YAML値と一致する必要があり、キャッシュ返却前と
直接calculate前の両方で確認する。CP2K parametersには`spin`を追加しない。
PySCF YAMLのspin=2Sとは区別する。ASE 3.28.0の`calculate()`はnumbers変化でだけ
force envを解放する(cp2k.py:293–297)。infoから状態を更新しない設計で、
古いforce envの電子状態を黙って再利用することを防ぐ。

ASE 3.28.0のcp2k.py:536はcommandに`cp2k_shell`を要求するが、PyPI最新の
3.29.0 wheelの同ファイル:533–539ではassertが削除されていた。
kitは3.29.0未満でだけこの検査を行い、依存範囲・constraintsは維持する。
生成時点でプロセスが起動するため`with`または`close()`が必要。
`free_energy`は`energy`と同値で、スメアのE0/自由エネルギーを区別できない。
ASEは常にGET_STRESSするため、応力を無効にすると0またはshellエラーになりうる。
部分周期スラブには雛形で`PERIODIC XY`と対応するPoisson solverを指定する。
詳しい制約と雛形は[CP2Kガイド](cp2k.md)を参照。
実機CP2K検証はTSUBAMEで別途行う。速いテストは偽の計算機と偽shellを使い、
slow H2テストは実行ファイル/データのテスト用環境変数が無ければskipする。
