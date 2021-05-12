# hw3
根據題意，我們得知有gesture_UI和tilt_angle兩個模式。因此先將RPCfunction各自連到一個Thread，再用各自的Thread分別控制一個function。

流程+結果：
  先確認在python執行的terminal上會顯示與mbed連結到同一網路，在cpp的terminal執行screen的指令也會先顯示有連結到同一ip。
  首先先用MQTT將PC和MBED做連結並顯示於螢幕，再經由USER輸入RPC指令(/g/run)啟動gesture_UI的模式(同時顯示LED1代表)，開始偵測手勢。當每次偵測到LAB8任何一項手勢，即開始將門檻角度增加五度，起始值為30度，
並顯示於液晶螢幕與與screen上。
  接著，當按下USER_BUTTON，則確認了門檻角度，將這則訊息同時顯示於液晶螢幕與與screen上，也用MQTT傳至另一個terminal做顯示
並停止偵測手勢。
  當USER輸入RPC指令(/a/run)，呼叫傾斜板子模式(同時顯示LED2代表)，則開始偵測角度，以靜止時的加速度和傾斜的加速度做內積得到cos，做每100ms顯示於screen上。
  若傾角超過USER自己所設定的門檻角度，則用MQTT回傳到另一個用python執行的terminal上，每超過就回傳一個值，直到10個角度(此為我所預設的)就會終止偵測角度模式。
