# Semantic-Segmentation-Materials

### Convolutional Neural Networks, CNNs
卷積神經網絡 (CNN) 已經被廣泛地應用於各種機器視覺的任務，例如圖像分類(image classification)、物件辨識(object detection)和語義分割(semantic segmentation)。CNN 通常是由多個卷積層組成，然後再接一個或多個全連接層。其中，卷積層主要是負責提取輸入樣本的空間特徵，而全連接層則是負責將這些特徵組合成最終的預測。


### Fully Convolutional Networks, FCNs
相對於CNN，FCN主要是為了語義分割而設計的目標是為圖像中的每個向素分配一個標籤，從而實現精確的識別物體並定位其邊界。FCN與CNN的差異在於其後段的全連結層會以卷積層取代，前段的卷積層和CNN一樣負責取得輸入樣本的空間特徵，而後段的卷積層則是負責上採樣以產生最終的分割結果。

![](https://i.imgur.com/X82zO1O.png)

### U-Net
但是在後段上採樣之前由於已經經過前段的下採樣，因此勢必會損失一些空間資訊。因此，Unet便在上採樣的過程中參考下採樣之前的資訊，從而保留更多有關輸入樣本的空間資訊。相較於FCN，Unet更有利於分割具有不規則形狀和小細節的圖像，經常被用於解決生物方面的醫學挑戰，而FCN則是具有更高的彈性進行各式各樣的應用。為了能夠更方便的取用，兩者的程式碼皆在[FCN.py](#code)與[U-Net.py](#code)中保存，並以Oxford-IIIT pet dataset作為個案演示。

![](https://i.imgur.com/IzZWZi0.png)

U2Net為U-Net的改進版本，透過ResNet架構使其能夠將不同級別的特徵進行彙整。

![image](https://user-images.githubusercontent.com/98240703/229375336-56361afa-991f-4e55-9ed1-d8bb5ec0ecb8.png)

