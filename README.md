ResNet-164[[1](https://arxiv.org/abs/1603.05027)]の、Kerasでの実装です。たぶん、もっともシンプルなコードなんじゃないかと。

# Usage

## 準備

~~~ bash
$ pip3 install --upgrade tensorflow-gpu keras funcy matplotlib h5py
~~~

## 訓練

~~~ bash
$ python3 train.py
~~~

## 訓練結果の確認

~~~ bash
$ python3 check.py
~~~

![loss](./results/loss.png)

![accuracy](./results/accuracy.png)

私が試した結果だと、CIFAR-10の精度は94.70%になりました。論文の94.54%に近い値なので、多分コードは大丈夫。

# Notes

* ごめんなさい。TensorFlow環境でしか試していません。
* [https://github.com/nutszebra/residual_net](https://github.com/nutszebra/residual_net)を参考にして作成しています。私の能力だと、論文だけでは作れませんでした……。
* Kerasに関数型プログラミングのテクニックを適用する方法は、[Kerasと関数型プログラミングで、楽ちんに深層学習してみた](https://tail-island.github.io/programming/2017/10/13/keras-and-fp.html)にまとめました。

# References

* Identity Mappings in Deep Residual Networks [[2](https://arxiv.org/abs/1603.05027)]
