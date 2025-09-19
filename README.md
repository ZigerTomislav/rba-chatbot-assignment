# Muzejski Chatbot - RBA Zadatak

Ovo je kratki sažetak mojeg rješenja zadatka
Link na github zadatka: https://github.com/ivandoslic/rba-chatbot-assignment

## Početak

Prvo sam stvorio [util.py](app/util.py) i klasu ChatBotService kojom sam si olakšao pozive na backend i lokalno sam pokrenuo app. Dodao sam i DataHandler klasu s kojom manipuliram testnim podacima koje sam spremio u .csv datoteke.

## Testiranje

Testiranje sam radio u jupyter notebooku [testing_report](app/testing_report.ipynb) gdje se detaljnije
vide greške koje sam pronašao.

Za početak sam složio svoj testni skup [dataset_test_hard.csv](app/data/dataset_test_hard.csv) koji je bio malo pretežak i model je imao samo 22% točnih predikcija. Proučio sam malo model i train skup i složio testni skup [dataset_test_easy.csv](app/data/dataset_test_easy.csv) sličniji train skupu i dobio bolji rezultat 77.27%

### Analiza grešaka
Detaljno sam u jupyter notebooku dao za svaku grešku mišljenje, ali ukratko mislim da je glavni razlog za većinu grešaka premali train skup podataka, dobar primjer je riječ "cijena" koja je samo u train skupu za "ulaznice" tako da ako pitamo za cijenu "članstva" ili cijenu "parkinga" model će misliti da se radi uvijek o ulaznicama (to je i bila jedna od grešaka). Mislim da je najslabija točka modela premali train dataset, nakon što bi povećali dataset mislim da još bismo dio grešaka mogli smanjiti sa boljim modelom ili kombinacijom više njih.


## Poboljšanje modela

U bilježnici [bot_improvement](app/bot_improvement.ipynb) sam probao koliko stignem dokazati svoje tvrdnje i poboljšati bota. 

### Povećanje train skupa
Pa sam krenuo prvo s pretpostavkom da je premali train skup, na originalni train skup sam nadodao svoj dataset_test_hard i testirao na dataset_test_easy te također obrnuto. U oba slučaja sam dobio bolji rezultat sa 22% na 41.27% i u drugom slučaju sa 77.27% na 86.33%. Tako da mislim da ovo potvrđuje da je train skup premalen.

### Data augmentation
Sljedeće što sam htio probati je proširenje podataka(data augmentation). Napravio sam klasu DataAugmenter
koja ima metode koje preuređuju train skup. Ideja mi je bila oponašati korisnike pa sam htio da model trenirati na podacima kojima su maknuti hrvatski znakovi "ć,č,đ,ž" i zamijenjeni sa "c,c,d,z", te sam još htio probati ako recimo neko prebrzo tipka pa zaboravi jedno slovo u rečenici. Iako sam ovako povećao train skup sa takvim podacima, nisam postigao bolje rezultate.

### Promjena modela
Za kraj sam još koliko sam stigao probao druge modele hoće li dati bolje rezultate. Pa sam testirao sa modelima xgboost, svm i randomForest, ali nisam dobio bolje rezultate.

---
<br>
<br>




To je ukratko što sam stigao u ova 3 dana odraditi, ako Vas zanimaju detalji možete ih pogledati u bilježnicama: [testing_report](app/testing_report.ipynb) i [bot_improvement](app/bot_improvement.ipynb) 



