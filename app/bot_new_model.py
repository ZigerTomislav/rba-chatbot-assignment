from typing import List, Dict, Tuple
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np
import util
import xgboost as xgb


# Kanonski odgovori po namjeri (HR)
CANONICAL: Dict[str, str] = {
    "radno_vrijeme": "Otvoreni smo uto–ned 10:00–18:00. Ponedjeljkom zatvoreno.",
    "ulaznice": "Opća ulaznica iznosi 12 €, studenti/umirovljenici 8 €, djeca do 12 g. besplatno.",
    "adresa": "Gradski Muzej Rivertown, Riječka avenija 12, Zagreb.",
    "danas_izlozbe": "Današnji naglasci: 'Impresionisti Save' i 'Stari obrti Slavonije'.",
    "kafic": "Kafić Riverbend u prizemlju nudi grickalice, kavu i čaj.",
    "toaleti": "Toaleti su na svakom katu pokraj dizala.",
    "pristupacnost": "Osiguran je pristup za invalidska kolica, dizala i besplatan ulaz za pratnju.",
    "parking": "Podzemna garaža je odmah do muzeja; prvi sat je besplatan.",
    "clanstvo": "Članarina od 45 € godišnje – besplatan ulaz i pozivnice za događaje.",
    "kontakt": "Kontakt: +385 1 555 123, hello@rivertownmuseum.hr.",
}


@dataclass
class ModelBundle:
    pipeline: Pipeline
    encoder: LabelEncoder
    labels: List[str]

_bundle: ModelBundle = None


def _build_pipeline(X = None, y = None) -> ModelBundle:

    if X is None or y is None:
        data_handler = util.DataHandler()
        X, y = data_handler.get_train_data_xy()

    print("making new pipeline, x:", len(X), "y:", len(y))

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)

    xgb_model = xgb.XGBClassifier(
        eval_metric='mlogloss',
        objective='multi:softprob',
        #objective=multi:softmax,
        n_estimators=10,
        max_depth=3,
        learning_rate=0.3,
        random_state=42,
        verbosity=1,

        #regularizacija za isprobat
        #subsample=0.8,
        #colsample_bytree=0.8

        reg_alpha=1.0,
        reg_lambda=1.0
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            analyzer="word",
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", xgb_model),
    ])
    pipe.fit(X, y_enc)

    return ModelBundle(pipeline=pipe, encoder=enc, labels=list(enc.classes_))


def ensure_model() -> ModelBundle:
    global _bundle
    if _bundle is None:
        _bundle = _build_pipeline()
    return _bundle


def predict(message: str) -> Dict[str, any]:
    b = ensure_model()
    probs = _predict_proba(b, [message])[0]
    top_idx = int(np.argmax(probs))
    intent = b.labels[top_idx]
    conf = float(probs[top_idx])
    reply = CANONICAL[intent]
    probs_named = {label: float(probs[i]) for i, label in enumerate(b.labels)}

    return {
        "intent": intent,
        "confidence": round(conf, 2),
        "reply": reply,
        "probs": probs_named,
        "trace": {
            "vectorizer": "tfidf(word 1-2gram)",
            "classifier": "xgboost",
            "language": "hr",
        }
    }


def _predict_proba(bundle: ModelBundle, texts: List[str]) -> np.ndarray:
    clf = bundle.pipeline.named_steps["clf"]
    tfidf = bundle.pipeline.named_steps["tfidf"]
    X = tfidf.transform(texts)
    return clf.predict_proba(X)


def list_intents() -> List[Dict[str, any]]:
    return [
        {
            "intent": i,
            "canonical_reply": CANONICAL[i],
            "examples": [t for t, lbl in TRAIN if lbl == i]
        }
        for i in CANONICAL
    ]