from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import matplotlib.pyplot as plt
# -------------pytorch----------------------
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""
したいこと➡LassoとLightGBMの特徴量の絞り込みの比較

Lassoでの特徴量の絞り込みとその数と同数のLightGBMでの絞り込みを行い、
それらと相関が少ないPytorchで実際に学習することで絞り込み制度の比較を行う。

LassoのAlpha値を徐々にあげていくことで特徴量を増やしていく。
その特徴量と同数のLightGBMの特徴量寄与度のデータから寄与度高い順に選ぶ。
Lassoで選ばれた特徴量とそれと同数のLightGBMで選ばれた特徴量
Pytorchで学習し評価を出す。
"""


# R2から2乗を無くして絶対値にすることではずれ値に強くする。
def R2_score_custom(y_true, y_pred):
    u = np.abs(y_true - y_pred).sum()
    v = np.abs(y_true - y_true.mean()).sum()
    R2 = 1 - (u / v)
    return R2


# Pytorchで最終予測
def pytorch_l(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # 専用の型に変更
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train.reshape((y_train.shape[0], 1))).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test.reshape((y_test.shape[0], 1))).float()
    # Tensorデータセットとしてまとめる
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_test, y_test)
    batch_size = 100
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # モデルの定義
    # 近年精度が高いと噂のMishの導入
    def mish(x):
        return x * torch.tanh(F.softplus(x))

    # モデル本体の定義 3層
    # input➡500➡10➡output
    class LinearRegression(nn.Module):
        def __init__(self, input_size, output_size):
            super(LinearRegression, self).__init__()
            self.fc1 = nn.Linear(input_size, 500)
            self.fc2 = nn.Linear(500, 50)
            self.fc3 = nn.Linear(50, output_size)

        def forward(self, x):
            x = self.fc1(x)
            x = mish(x)
            x = self.fc2(x)
            x = mish(x)
            x = self.fc3(x)
            return x

    net = LinearRegression(input_size=x_train.shape[1], output_size=1)
    criterion = nn.MSELoss()  # 評価関数の設定
    optimizer = optim.Adam(net.parameters(), lr=0.01)  # optimizerの設定(Adam)

    # 学習・評価
    # エポック数
    num_epochs = 50
    dataloaders_dict = {"train": train_dataloader, "val": valid_dataloader}

    for epoch in range(num_epochs):
        # 学習と評価の切り替え
        for phase in ["train", "val"]:
            if phase == "train":
                net.train()  # モデルを学習モードに設定
            else:
                net.eval()  # モデルを評価モードに設定

            # 損失和
            epoch_loss = 0.0
            # データ数が少ないのでバッチ毎ではなくエポック毎に結合してデータ数を少し多くしてから評価する
            y_pred = np.empty(0)
            y_true = np.empty(0)
            # DataLoaderからデータをバッチごとに取り出す
            for inputs, labels in dataloaders_dict[phase]:

                # optimizerの初期化
                optimizer.zero_grad()

                # 学習時のみ勾配を計算させる設定にする
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)  # 予測値
                    loss = criterion(outputs, labels)  # 損失を計算
                    # データ数が少ないのでバッチ毎ではなくエポック毎に結合してデータ数を少し多くしてから評価する。
                    y_pred = np.append(y_pred, outputs.to("cpu").detach().numpy().copy().flatten())
                    y_true = np.append(y_true, labels.to("cpu").detach().numpy().copy().flatten())
                    # 訓練時はバックプロパゲーション
                    if phase == "train":
                        # 逆伝搬の計算
                        loss.backward()
                        # パラメータの更新
                        optimizer.step()
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)

            # # epochごとのlossと正解率を表示
            # epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            # R2 = R2_score_custom(y_true=y_true, y_pred=y_pred)
            # print(f"{epoch}-{phase} MSE:{epoch_loss:.5f} - R2:{R2:.3f}")
            if epoch == (num_epochs - 1) and phase == "val":
                return y_true, y_pred


# 特徴量の係数・寄与度を一律全体の何パーセントかに切り替え、マイナスに関しても絶対値で消去
def importance_choose(imp_data):
    Abs = np.abs(imp_data)
    Sum = np.sum(Abs)
    r = Abs / Sum
    return r


# データの準備
import mglearn

X, y = mglearn.datasets.load_extended_boston()
print("X", X.shape)
from sklearn.model_selection import KFold
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
# Lasso LightGBM 特徴量の係数・寄与度の相関性

# LASSO
st = time.time()
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print(f"Lasso_time:{(time.time()-st)}")
coef_idx = np.where(lasso001.coef_ != 0)[0]  # 係数が0の特徴量を抽出
new_X = X[:, coef_idx]  # 係数が0以外の特徴量を選択

# 2．LGB
import lightgbm as lgb
params = {"boosting_type": "gbdt", "objective": "regression", "metric": "rmse", "verbose": -1,
          "min_data_in_leaf": 10, "num_leaves": 10}  # データ数が少ないことに合わせて設定
# LGB用のデータセットの作成
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# LGBの学習
st = time.time()
model = lgb.train(params, lgb_train, valid_sets=lgb_eval, verbose_eval="False", num_boost_round=200)
print(f"LGB_time:{(time.time()-st)}")
lgb_l1_fi = model.feature_importance(importance_type="gain")  # 特徴量の係数を寄与度方式で取得
lgb_l1_fi = importance_choose(imp_data=lgb_l1_fi)  # 寄与度を絶対値かつ割合で算出
lasso001_fi = importance_choose(imp_data=lasso001.coef_)  # 寄与度を絶対値かつ割合で算出

plt.plot(lgb_l1_fi)
plt.plot(lasso001_fi)
plt.show()
num = len(coef_idx)  # 係数が0ではない特徴量の数
feature_importance_idx = np.argsort(lgb_l1_fi)[::-1][:num]
# 特徴量の一致度を検索
num_and = len(set(coef_idx) & set(feature_importance_idx))
print(f"一致度:{num_and}/{num}")
"""
一致度や図から見ての通り相関性は低い。
お互いにあんまり相関性が見られない。
"""
# 交差検証 / 分割数4
n_split = 4
kf = KFold(n_splits=n_split, shuffle=True)
# 各スコアと特徴量数のログ保存用リスト
score_lgbs = []
score_lassos = []
feature_nums = []

# LassoのAlpha値 設定用 / 0.01 * 1~100
alpha_list = [i for i in range(1, 100, 10)]
for alpha in alpha_list:
    y_pred_lassos = np.empty(0)
    y_pred_lgbs = np.empty(0)
    y_true_lassos = np.empty(0)
    y_true_lgbs = np.empty(0)
    all_feature_num = 0
    for idx, (train, test) in enumerate(kf.split(X)):
        # 各種データ分割
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]

        # 標準化
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        scaler.fit(X_test)
        X_test = scaler.transform(X_test)

        # 1．LASSO
        alpha2 = 0.01 * alpha
        lasso001 = Lasso(alpha=alpha2, max_iter=100000).fit(X_train, y_train)
        coef_idx = np.where(lasso001.coef_ != 0)[0]  # 係数が0の特徴量を抽出
        new_X = X[:,coef_idx] # 係数が0以外の特徴量を選択
        y_true_lasso, y_pred_lasso = pytorch_l(new_X, y)  # pytorchで予測 / 戻り値は正解ラベルと予測値
        # データ数が少ない為、あえて評価せずに交差検証全データをまとめて評価する
        y_pred_lassos = np.append(y_pred_lassos, y_pred_lasso)
        y_true_lassos = np.append(y_true_lassos, y_true_lasso)

        # 2．LGB
        import lightgbm as lgb

        params = {"boosting_type": "gbdt", "objective": "regression", "metric": "rmse", "verbose": -1,
                  "min_data_in_leaf": 10, "num_leaves": 10}  # データ数が少ないことに合わせて設定
        # LGB用のデータセットの作成
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # LGBの学習
        model = lgb.train(params, lgb_train, valid_sets=lgb_eval, verbose_eval="False", num_boost_round=200)
        lgb_l1_fi = model.feature_importance(importance_type="gain")  # 特徴量の係数を寄与度方式で取得
        lgb_l1_fi = importance_choose(imp_data=lgb_l1_fi)  # 寄与度を絶対値かつ割合で算出
        lasso001_fi = importance_choose(imp_data=lasso001.coef_)  # 寄与度を絶対値かつ割合で算出

        num = len(coef_idx)  # 係数が0ではない特徴量の数
        all_feature_num += num
        feature_importance_idx = np.argsort(lgb_l1_fi)[::-1][:num]  # 係数が大きい物から降順に並び替えた後、lassoと同数の特徴量に係数が大きい物から選ぶ
        new_X = X[:, feature_importance_idx]  # 特徴量を選択
        y_true_lgb, y_pred_lgb = pytorch_l(new_X, y)  # pytorchで予測 / 戻り値は正解ラベルと予測値
        # データ数が少ない為、あえて評価せずに交差検証全データをまとめて評価する
        y_pred_lgbs = np.append(y_pred_lgbs, y_pred_lgb)
        y_true_lgbs = np.append(y_true_lgbs, y_true_lgb)

        # 特徴量の一致度を検索
        num_and = len(set(coef_idx) & set(feature_importance_idx))

    # 評価
    from sklearn.metrics import mean_squared_error

    score_lgb = mean_squared_error(y_true_lgbs, y_pred_lgbs)
    score_lasso = mean_squared_error(y_true_lassos, y_pred_lassos)
    print("lgb評価:", score_lgb)
    print("las評価:", score_lasso)

    score_lgbs.append(score_lgb)
    score_lassos.append(score_lasso)
    feature_nums.append(all_feature_num / n_split)

# 各線グラフの色を指定
color1 = "b"
color2 = "r"
color3 = "g"

# グラフ描画領域を作成
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)  # 1つ目の描画領域
ax2 = ax1.twinx()  # 1つ目の描画領域とx軸を共有する描画領域を作成

# 線グラフを描画
ax1.plot(alpha_list, score_lgbs, color=color1, label="score_lgb")
ax1.plot(alpha_list, score_lassos, color=color2, label="score_lasso")
ax2.plot(alpha_list, feature_nums, color=color3, label="Feature_num")

# 各描画領域の凡例の情報を取得
handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()

# y軸の色を変更
ax2.spines["left"].set_color(color1)
ax2.spines["right"].set_color(color3)

# 凡例を右上に余白無し
ax1.legend(
    handler1 + handler2,
    label1 + label2,
    loc="upper right",
    borderaxespad=0,
)

# y軸の値の色を変更
ax1.tick_params(axis="y", colors=color1)
ax2.tick_params(axis="y", colors=color2)

# 軸ラベルを設定
ax1.set_ylabel("Score")
ax2.set_ylabel("Feature_num")

plt.show()  # グラフを描画

"""
Lassoの方が基本的にLightGBMより計算が早い。
また特徴量の係数も適切で特徴量選択にも優れている。
特徴量選択においてLassoの有用性が見られた。

"""
