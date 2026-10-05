# 📦 Demand Forecasting & Supply Chain Optimization
### Prévision de la Demande Multi-Points de Vente & Optimisation Logistique

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🎯 Contexte & Problématique Métier

Dans le secteur de la distribution et du retail, la planification précise des réapprovisionnements est un levier financier stratégique direct :
* **Le sous-stockage (rupture)** génère une perte sèche de chiffre d'affaires, une détérioration de l'image de marque et une insatisfaction client.
* **Le sur-stockage** immobilise inutilement de la trésorerie (BFR), engendre des coûts de possession d'entrepôt et des risques de dépréciation/obsolescence.

**Le Défi :** Prédire avec précision la demande quotidienne à horizon **90 jours** pour **50 références de produits** réparties sur **10 points de vente** (500 séries temporelles distinctes observées sur 5 ans), afin d'éclairer les décisions d'achat et d'optimiser les stocks tampons.

---

## 🏗️ Architecture du Projet

```
demand-forecasting/
├── data/
│   ├── raw/                  # Données brutes (train.csv, test.csv)
│   └── processed/            # Features calculées & prédictions finales
├── notebooks/
│   ├── 01-eda.ipynb          # Exploration temporelle, saisonnalités, distribution
│   ├── 02-feature-engineering.ipynb # Extraction des lags, fenêtres glissantes (anti-leakage)
│   └── 03-modelling.ipynb    # Validation temporelle, Baseline naïve, LightGBM, ROI
├── src/
│   ├── __init__.py
│   └── app.py                # Dashboard interactif Streamlit d'aide à la décision
├── requirements.txt
└── README.md
```

---

## 🔬 Démarche Méthodologique

### 1. Analyse Exploratoire des Données (EDA)
* **Intégrité :** 500 séries temporelles continues (2013-2017) sans valeurs manquantes.
* **Composantes temporelles :**
  * Tendance de fond régulière de +8 à +10 % par an.
  * Forte saisonnalité intra-annuelle (creux en fév., pic en juillet).
  * Effet jour de semaine marqué (croissance progressive du lundi au pic du dimanche).
* **Hétérogénéité :** Disparités importantes entre magasins (Store 2 & 8 sur-performants) et entre produits (ratio de 4,5x entre best-sellers et références secondaires).

### 2. Feature Engineering & Prévention Strict du Data Leakage
Comme l'horizon de test est de **90 jours**, aucune variable ne doit utiliser d'information ultérieure à J-90 :
* **Variables de calendrier :** `dayofweek`, `month`, `year`, `dayofyear`, `is_weekend`, `is_month_start/end`.
* **Variables de retard (Lags) :** Décalages de sécurité à **91, 98, 105, 112, 119, 126, 182 et 364 jours** pour capter la saisonnalité annuelle et hebdomadaire passée sans fuite d'information.
* **Moyennes mobiles (Rolling Features) :** Moyennes sur 7, 14, 30 et 90 jours, ainsi que volatilité (écart-type sur 30j).
* **Normalisation de la cible :** Transformation $\log(1 + x)$ pour symétriser la distribution des ventes.

### 3. Stratégie de Validation & Modélisation
* **Split Temporel Out-of-Time :** Évaluation sur les 3 derniers mois de 2017 (Octobre à Décembre 2017, soit 46 000 points de test) pour simuler exactement les conditions réelles de prédiction.
* **Baselines Métier :**
  1. *Baseline Naïve (Lag 364j) :* Prévision basée sur les ventes de l'année précédente à la même date.
  2. *Baseline Moyenne Mobile (30j) :* Prévision basée sur la moyenne glissante récente.
* **Modèle Champion :** Gradient Boosting via **LightGBM**, optimisé avec early stopping et régularisation des feuilles.

---

## 📊 Résultats & Performances Comparées

| Approche | Type | MAE (unités) | RMSE (unités) | SMAPE (%) | WAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Moyenne Mobile (30j)** | Baseline statistique | 15.83 | 19.88 | 27.99 % | 28.96 % |
| **Ventes Année N-1 (Lag 364j)** | Baseline métier | 8.35 | 10.90 | 17.86 % | 15.28 % |
| **LightGBM Tuné** | **Modèle IA Final** | **5.89** | **7.66** | **12.41 %** | **10.77 %** |

> 🚀 **Gain :** Le modèle LightGBM réduit l'erreur de prévision de **plus de 5,4 points de SMAPE** et de **près de 30 % en erreur absolue unitaire** par rapport à la méthode métier historique.

---

## 💼 Impact Métier & ROI Chiffré

Afin de traduire la performance technique en valeur concrète pour la Direction Financière et la Supply Chain, un modèle de coût logistique a été appliqué sur la période de test :
* **Hypothèse Marge brute unitaire perdue (Rupture) :** 15,00 €
* **Hypothèse Coût de possession hebdomadaire (Surstock) :** 2,00 €

```
============================================================
BILAN FINANCIER SIMULÉ SUR 3 MOIS (10 Magasins, 50 Produits)
============================================================
Coût lié aux erreurs de prévision (Méthode Naïve N-1) : 3 940 852 €
Coût résiduel avec les prévisions LightGBM           : 2 583 480 €
------------------------------------------------------------
ÉCONOMIE ESTIMÉE (selon les hypothèses ci-dessus)  : 1 357 372 €
RÉDUCTION DES PERTES FINANCIÈRES LOGISTIQUES         : -34.4 %
============================================================
```

---

## 🖥️ Livrable Applicatif : Dashboard Décisionnel Streamlit

Une interface interactive a été développée dans [`src/app.py`](src/app.py) pour permettre aux gestionnaires d'approvisionnement :
1. De filtrer par magasin et par référence produit.
2. De visualiser la projection des ventes à 90 jours avec comparaison de l'historique récent.
3. De disposer d'un calcul automatique du **seuil de réassort / stock tampon recommandé**.
4. De simuler en temps réel le ROI financier selon les marges et coûts de stockage de l'entreprise.

### Lancement de l'application :

```bash
# 1. Cloner le dépôt et créer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Récupérer les données (non versionnées dans le dépôt)
#    Kaggle : "Store Item Demand Forecasting Challenge"
#    Placer train.csv et test.csv dans data/raw/

# 4. Exécuter les notebooks dans l'ordre : 01 -> 02 -> 03
#    (le notebook 03 génère data/processed/predictions_test.csv)

# 5. Lancer le dashboard interactif
streamlit run src/app.py
```

---

## 🛠️ Compétences Démontrées

* **Machine Learning & Time Series :** LightGBM, GBDT, Feature Engineering temporel, validation out-of-time stricte, métriques SMAPE / WAPE.
* **Consulting & Data Storytelling :** Cadrage du problème business, baselines de comparaison, conversion d'erreurs statistiques en ROI financier (€).
* **Data Engineering & Product :** Pipeline de données reproductible, architecture modulaire, dashboard interactif de restitution.
