# 俺らはサンタさん作ってんだよ！

## christmas2018
これが性（情報系）の6時間
+ GANを用いたサンタ生成プロジェクト

## Progress
+ 2時間程度でデータ収集とネットワークの構成まで終わった
+ 学習が上手くいかなかった
+ 寝た

## Folder
christmas2018/data/santa に学習データをいれる

    git clone https://github.com/gojirokuji/christmas2018.git
    cd christmas2018/
    mkdir data generated models
    mkdir data/santa

## Result
Generated images:
![Generated images](https://raw.githubusercontent.com/gojirokuji/christmas2018/master/images/gs.jpg)

Train 200 epochs:
![Loss plot](https://raw.githubusercontent.com/gojirokuji/christmas2018/master/images/trainingLossPlot.png)

データ拡張を行い学習データをかさ増ししたが、回転は行わない方が良かったと思われる。また、データ拡張 or Mode Collapseが原因と考えられるが、似たような画像が多く生成されてしまった。

## Special Thanks
[TatsukiSerizawa (Sweater)](https://github.com/TatsukiSerizawa)
