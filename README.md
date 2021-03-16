# simple-virtual-mouse-using-mediapipe
MediaPipeを用いたハンドジェスチャーによる簡単なマウス操作を行うプログラムです。<br>
マウス移動：手の移動<br>
マウス左クリック：手をパーからグーに変える<br><br>
<img src="https://user-images.githubusercontent.com/37477845/111347926-070d7800-86c3-11eb-851f-22f1e87f67b5.gif" width="75%">

# Requirements
* mediapipe 0.8.1 or Later
* PyAutoGUI 0.9.52 or Later
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later

# Demo
Webカメラを使ったデモの実行方法は以下です。
```bash
python app.py
```

デモ実行時には、以下のオプションが指定可能です。
* --device<br>カメラデバイス番号の指定 (デフォルト：0)
* --width<br>カメラキャプチャ時の横幅 (デフォルト：960)
* --height<br>カメラキャプチャ時の縦幅 (デフォルト：540)
* --min_detection_confidence<br>
検出信頼値の閾値 (デフォルト：0.7)
* --min_tracking_confidence<br>
トラッキング信頼値の閾値 (デフォルト：0.5)
* --margin_width<br>ハンドジェスチャーの操作範囲(横幅のマージン)(デフォルト：0.2)
* --margin_height<br>ハンドジェスチャーの操作範囲(縦幅のマージン)(デフォルト：0.2)

# Reference
* [MediaPipe：Hands](https://google.github.io/mediapipe/solutions/hands)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
simple-virtual-mouse-using-mediapipe under [Apache-2.0 License](LICENSE).
