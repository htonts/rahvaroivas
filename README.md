# **Kasutusjuhend rahvarõivaste tuvastamise programmile**

1. Nõuded

   1.1. Kiirem treenimine GPU abil
   
2. Failide kirjeldused
      
    2.1. augment.py     

    2.2. training.py 

    2.3. prediction.py
   
    2.4. idenprof kaust

3. Mudelite treenimine
    
    3.1. Ettevalmistus

    3.2. Treenimine

4. Viited

## **1. Nõuded**

- Python 3.7.6
- Tensorflow 2.0.4
- Keras 2.4.3
- Numpy 1.18.5
- Pillow 7.0.0
- Scipy 1.4.1
- H5py 2.10.0
- Matplotlib 3.3.2
- Opencv-python
- Keras-resnet 0.2.0
- ImageAI 2.1.6

**Käsud kõikide vajalike komponentide installimiseks:**

pip install tensorflow==2.0.4

pip install keras==2.4.3 numpy==1.18.5 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0 

pip install imageai

### **1.1. Kiirem treenimine NVIDIA GPU abil**

Tensorflow 2.4.0 võimaldab kiiremat treenimist NVIDIA GPUde abil, kasutades CUDA paralleelarvutuse platformi. 
Võttes näiteks ImageAI näidismaterjali, vähenes testarvutil mudelite treenimisaeg seitsmelt päevalt neljale tunnile.

Tensorflow 2.4.0 tõttu ei ole võimalik igas arvutis treenida ega ennustada treenitud mudeli abil, mistõttu on põhinõudena
välja toodud Tensorflow **2.0.4**.

Nõuded:

- NVIDIA graafikakaart (Minimaalselt GTX 1030 mudel)
- CUDA 11.0 ja 11.1
- cuDNN 8.1.0
- Tensorflow 2.4.0

Õpetus:
1. Tõmmata Nvidia CUDA arhiivist nii CUDA 11.0 kui ka CUDA 11.1. (https://developer.nvidia.com/cuda-toolkit-aRCHIVE)
2. Tõmmata Nvidia Developer arhiivist cuDNN 8.1.0 (https://developer.nvidia.com/CUDnn, vajab Nvidia kasutajat)
3. Installida nii CUDA 11.0 kui ka CUDA 11.1
4. Peale installi kopeerida cuDNN 8.1.0 sisu mõlemasse CUDA installatsiooni kausta 
   (tavaliselt C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0)
   
5. Kopeerida v11.1\bin alt **ptxas.exe** v11.0\bin kausta ning asendada.
6. Käivitada treenimise fail.

## **2. Failide kirjeldused**

### **2.1. augment.py**

Augment.py käivitamise tulemuseks on Augmentori abil kunstlikult suurendatud hulk testimis- ja treenimismaterjale
tulevase mudeli jaoks.

#### **2.1.1. Augmentor**

Augmentor on pildimoonutusmoodul, mille abil saame väiksemat materjalikogust tehislikult rohkendada.

Augmentoril on mitmeid valikuid pildimoonutuseks, mida kõiki saab kombineerida, kuid põhilisena toon välja järgmised:
- Elastiline moonutamine
- Perspektiivimoonutused
- Pildi suurust säilitavad pööramised
- Pildi suurust säilitavad lõikamised
- Kärpimine

#### **2.1.2. augment.py tööprotsess**

Klassi käivitamisel kogutakse kokku kõik treenimiskaustas olevate kaustade ehk klassifikatsioonide nimed, eraldades 
tekkinud nimekirjast juurkausta ehk treenimiskausta.

Seejärel võetakse kõik kaustas olevad pildid ning rohkendatakse need etteantud parameetrite järgi.

Kuna augmenteerimisel teisaldatakse kõik tekkinud pildid uude kausta, liigutatakse viimase sammuna rohkendatud pildid
tagasi klassifikatsioonikausta, samale tasemele, kus olid originaalpildid. Sama kordub testimiskaustas asuvate
materjalidega.

Rohkendamise ajaline kestus kirjutatakse logifaili "augmentLogs.txt", mis asub "logs" kaustas.

### **2.2. training.py**

Training.py käivitamise tulemuseks on vastava materjali abil treenitud mudel ja klassifikatsioonide fail,
mille abil saame ennustada piltide kategooriat.

#### **2.2.1. Mudeli treenimine**

Mudeli treenimise osas määrame treenitava mudeli tüübi, treenimisel kasutatava materjali kausta ning mudeli treenimise
parameetrid, milleks on:

- num_objects (kohustuslik)
- num_experiments (kohustuslik)
- enhance_data (valikuline, vaikimisi True kui pilte üle 1000)
- batch_size (valikuline, vaikimisi 32)
- show_network_summary (valikuline, vaikimisi False)
- initial_learning_rate (valikuline)
- training_image_size (valikuline, vaikimisi 224, mitte väiksem kui 100)
- continue_from_model (valikuline)
- transfer_from_model (valikuline)
- transfer_with_full_training (valikuline)
- save_full_model (valikuline)

Tulemuseks on num_experiments parameetris defineeritud väärtuses mudeleid ja klassifikatsioonifail, mille abil saame
Prediction.py failis ennustada piltide kategooriat.

Treenimine ajaline kestus kirjutatakse logifaili "trainingLogs.txt", mis ilmub "logs" kausta.

### **2.3. Prediction.py**

Defineerides mudeli nime, testpiltide kausta ja klassifikatsioonide asukoha saame ennustada valitud pildi kategooriat. 
Tulemus kirjutatakse logifaili "results.txt", mis ilmub "logs" kausta. Seal on näha nii pildi nimi kui tulemused. 

### 2.4. Idenprof kaust

Siin kaustas asuvad treenimis- ja testimismaterjalid, mudelid ning klassifikatsioonide fail ImageAI loojate poolt
pakutud näidismaterjalist.

Idenprof kaust on jagatud neljaks:
- json - Siin kaustas asub klassifikatsioonide fail, kus on kirjas kategooriad, mida mudel ennustama peab.
- models - Siia tekivad treenimise käigus tekitatud mudelid.
- test - Siin asuvad testmaterjalid, mida ImageAI kasutab treenitud mudeli täpsuse testimiseks.
- train - Siin asuvad treenimismaterjalid, mille abil ImageAI treenib mudelit etteantud parameetrite alusel.

## **3. Mudelite treenimine**

### **3.1. Ettevalmistamine**

Mudelite treenimiseks on vaja ette valmistada treenimis- ja testimismaterjalid. Ettevalmistuseks on vaja tõsta pildid 
testimis- ja treenimiskaustadesse. Mõlemas kaustas tuleb pildid omakorda jagada eri kaustadesse,
mille põhjal mudel saab ennustamise klassifikatsioonid.

### **3.2. Treenimine**

Treenimiseks tuleb teha järgmist:
1. (Valikuline) Augmentori abil testimis- ja treenimismaterjale tehislikult täiustada;
2. Määrata treenitava mudeli tüüp (MobileNetV2, ResNet50, InceptionV3, DenseNet121);
3. Määrata testimis- ja treenimismaterjalide ülemkaust;
4. Määrata treenimisparameetrid;
5. Käivitada training.py.

Olenevalt treenimiseks kasutatava arvuti tugevusest ning testimisel ja  treenimisel kasutatava materjali kogusest, 
võib treenimine võtta **mõnest minutist mõne kuuni.**


## **4. Viited**

1. Moses & John Olafenwa, ImageAI, A python library built to empower developers to build applications and systems 
   with self-contained Computer Vision capabilities https://github.com/OlafenwaMoses/ImageAI
2. Moses & John Olafenwa, Idenprof, A collection of images of identifiable professionals. 
   https://github.com/OlafenwaMoses/IdenProf
3. Marcus D. Bloice, Augmentor, Image augmentation library in Python for machine learning.
https://github.com/mdbloice/Augmentor