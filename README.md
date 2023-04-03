# 深度學習之語意分割Tensorflow/Pytorch模版

就如同標題所示，這裡將會以Tensorflow的影像資料集展示語意分割神經網路的模板，並對卷積系列的語意分割網路進行一些統整。

## 目錄

- [背景](#background)
- [相關技術說明](#install)
- [程式碼資源](#install)


## 背景

在近幾十年深度學習的研究進展中，卷積處理已經被證明在對空間上的特徵具有非常強健的處理能力，使神經網路能夠處理幾百萬個像素之間的潛在關係。過去ResNet(2015)、DenseNet(2017)的研究成果也使卷積神經網絡(CNN)的性能獲得大幅的成長，在我們能夠在最少的參數計算量下獲得最多的空間特徵。影像便是將空間的物理狀態量化成數位資訊的重要管道，透過極細小的光感元件將世界中的色彩轉化為RGB像素值(0~255, 0~255, 0~255)，並將這些數據統整到影像檔中紀錄。而針對這些影像的分析依照難度可以依序劃分成幾個等級： 1. 圖像分類 2. 語意分割 3.影像生成。

其中，語意分割是目前最受到醫學成像、機器人、自動駕駛學者非常關注的議題，藉由圖像(像素的集合)的某些特徵為圖像中的每個向素分配一個標籤，依此將圖像劃分成多個區域，使我們可以從畫面中精確的識別物體並定位其邊界。下個章節我們便會探討這些影像分割的技術原理與進展。

## 相關技術說明

### FCN
沿襲過去在**圖像分類**的成功案例，我們可以透過改變卷積操作的連結方式來取用這些特徵與資訊。FCN就是一個非常經典的例子，過去CNN的架構主要是在前段布置卷積層以擷取圖像特徵 **(編碼)**，後段則是透過全連結層(full connetion layer, FC-layer)將這些特徵進行分類 **(解碼)**。然而，FC-layer在數萬個像素預測值面前顯得，一個卷積操作就能抵上幾千個神經元的工作，也更能有效地處理非線性問題，因此FCN的概念便是將後段的FC-layer替換成Conv-layer，如下圖的連結方式。

 <img src="https://i.imgur.com/X82zO1O.png" width = "600"/>

### U-Net
但是在FCN後段解碼的過程中，由於已經經過前段的下採樣，因此勢必會損失一些空間資訊。因此，Unet便在上採樣的過程中參考在下採樣之前的資訊，從而保留更多有關輸入樣本的特徵資訊。相較於FCN，Unet更有利於分割具有不規則形狀和小細節的圖像，經常被用於解決生物方面的醫學挑戰，而FCN則是具有更高的彈性進行各式各樣的應用。

<img src="https://i.imgur.com/IzZWZi0.png" width = "700"/>

### U2-Net
U2-Net是一項較為新型的Unet，透過搭配多個Unet將不同量級的特徵進行彙整，並結合了自注意力機制，使其能夠選擇性地突出顯示重要的特徵與區域，實現更佳的分割精度與通用性，而參數量與運算資源便是它目前主要的限制。

<img src="https://user-images.githubusercontent.com/98240703/229375336-56361afa-991f-4e55-9ed1-d8bb5ec0ecb8.png" width = "500"/>


## 程式碼資源

|        模型     | `FCN`          |`UNet`            |`U2Net`                |
| :---:           | :---:            | :---:            | :---:            | 
| 檔名/版本         |  Oxford-IIIT pet dataset     |  Oxford-IIIT pet dataset    | Oxford-IIIT pet dataset    | 
| 檔名/版本         |  [FCN-keras.py]()     |  [Unet-keras.py]()    |- [U2Net-keras.py]()         | 

* FCN-keras.py

由於語意分割本來就是相對比較困難的任務，因此，直接利用keras硬train整個FCN將會非常困難，很容易overfitting或underfitting(視問題而定)。而較為穩健的做法便是將前段的編碼器與後段的解碼器分開來訓練，在這份檔案中則是先使用Tensorflow的預訓練模型VGG-16作為下採樣的工具，而後段則是沿用FCN的上採樣層，使模型能夠更有效的收斂。下圖為整個網路的架構。

<img src="https://user-images.githubusercontent.com/98240703/229553873-c7646ea4-dcb2-4b8d-96a9-a6d297bfed51.png" width = "700"/>

* Unet-keras.py


### 備註欄
def dice_loss(y_true, y_pred):
    y_true = tf.cast(tf.one_hot(indices=y_true, depth=num_classes), tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator

def IoU(y_true, y_pred, num_classes=3, smooth=1):
    y_true = tf.cast(tf.one_hot(indices=y_true, depth=num_classes), tf.float32)
    iou_scores = []
    for c in range(num_classes):
        true_class = y_true[..., c]
        pred_class = y_pred[..., c]
        intersection = tf.reduce_sum(tf.cast(true_class * pred_class, tf.float32))
        union = tf.reduce_sum(tf.cast(tf.math.logical_or(true_class, pred_class), tf.float32))
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou)
    return tf.reduce_mean(iou_scores)
    
 
Dice損失函, focal損失函,
Hausdorff°和 boundary損失函
