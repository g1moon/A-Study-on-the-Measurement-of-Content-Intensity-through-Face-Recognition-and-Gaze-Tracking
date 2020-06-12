# [KSC2019]A Study on the Measurement of Content Intensity through Face Recognition and Gaze Tracking(얼굴 인식과 시선 추적을 통한 콘텐츠 몰입도 측정 방안)


[[Paper](http://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE09301719&language=ko_KR)] [[Ppt](https://drive.google.com/open?id=1rMw_KV74t2mYoBclJsawVDMdCPEFsaPY)]

## 
정보 통신 기술의 발전에 따라 동영상 콘텐츠의 양이 폭발적으로 증가하고 있고, 교육의 목적과 방법
의 변화에 따라 다양한 교육 영상 콘텐츠의 관심이 증가하고 있다. 이러한 교육 환경의 변화에 맞게 몰 입도 있는 콘텐츠 제작 필요성이 증가하고 있고, 콘텐츠 몰입도 측정의 필요성이 부각되고 있다. 본 논 문에서는 콘텐츠 몰입도 측정을 위해 객관적인 방법으로 사용자의 인식과 함께 눈동자의 움직임을 이용 해 콘텐츠 몰입도 효과를 측정할 수 있는 방법을 제안한다. 콘텐츠를 보고 있는 것을 측정하기 위해 카 메라로 획득한 영상을 통해 얼굴을 검출하고, 검출된 정보를 바탕으로 사용자의 성별과 연령대를 측정한 다. 사용자 특성에 따라 콘텐츠의 시간대별 응시 여부와 응시 시간을 측정하여 성별과 연령대에 따라 객 관적인 콘텐츠 몰입도를 측정하는 기법을 제안한다.


## Environment 
- opencv : '3.4.5'
- dlib : '19.17.99'
- keras : '2.2.4'
- tensorflow : '1.14.0'
- numpy :  '1.16.1'

## Model
<img width="802" alt="스크린샷 2020-06-12 오후 4 24 38" src="https://user-images.githubusercontent.com/44131043/84476524-3f61cf00-acc9-11ea-86f5-c86fdf921715.png">
<img width="806" alt="스크린샷 2020-06-12 오후 4 24 31" src="https://user-images.githubusercontent.com/44131043/84476531-412b9280-acc9-11ea-9e69-f23d4ad75732.png">

## Evaluation
![loss](https://user-images.githubusercontent.com/44131043/84477236-49380200-acca-11ea-8340-7c92da1c48bd.png)

## Data download

###  IMDB-WIKI dataset 
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar

### UTKFace dataset

https://drive.google.com/open?id=0BxYys69jI14kSVdWWllDMWhnN2c

## Test
```python3 test.py```

<img width="619" alt="스크린샷 2020-03-23 오후 4 58 20" src="https://user-images.githubusercontent.com/44131043/77294369-84f4a880-6d27-11ea-869b-ab0aefda1765.png">


