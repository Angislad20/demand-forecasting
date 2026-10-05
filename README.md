# Demand Forecasting & Supply Chain Optimization
Prévision de la demande multi-magasins et optimisation des stocks

## Contexte et Objectif

Dans le secteur de la distribution, la gestion des approvisionnements repose sur un double arbitrage :
- Éviter les ruptures de stock qui entraînent des pertes directes de chiffre d'affaires et pénalisent la satisfaction client.
- Éviter le surstockage qui immobilise de la trésorerie et engendre des coûts d'entreposage.

L'objectif de ce projet est de prévoir les ventes quotidiennes à un horizon de 90 jours pour 50 produits répartis dans 10 magasins (500 séries temporelles sur 5 ans), afin d'anticiper les besoins et de calibrer les stocks de sécurité.

## Structure du Dépôt

```
demand-forecasting/
├── data/
│   ├── raw/                  # train.csv, test.csv (données brutes)
│   └── processed/            # train_features.csv, predictions_test.csv
├── notebooks/
│   ├── 01-eda.ipynb          # Exploration, saisonnalités et distributions
│   ├── 02-feature-engineering.ipynb # Variables de retard, moyennes mobiles et calendrier
│   └── 03-modelling.ipynb    # Baselines, modèle LightGBM et simulation financière
├── src/
│   ├── __init__.py
│   └── app.py                # Dashboard interactif Streamlit
├── requirements.txt
└── README.md
```

## Méthodologie

### 1. Analyse Exploratoire (EDA)
- Historique complet de 2013 à 2017 sans valeurs manquantes.
- Tendance haussière annuelle régulière (+8 à +10 % par an).
- Saisonnalité marquée : creux en janvier-février, pic en juillet, et hausse progressive du lundi au dimanche.

### 2. Feature Engineering (Prévention des fuites de données)
L'horizon de prévision étant de 90 jours, aucune variable n'utilise d'information à moins de 91 jours du point de prédiction :
- Variables calendaires : jour de la semaine, mois, jour de l'année, indicateurs de week-end et début/fin de mois.
- Variables de retard (Lags) : décalages de 91, 98, 105, 112, 119, 126, 182 et 364 jours pour capter la saisonnalité sans data leakage.
- Moyennes mobiles : fenêtres de 7, 14, 30 et 90 jours calculées à partir de la valeur à J-91.
- Transformation de la cible : passage en log1p pour symétriser la distribution des ventes.

### 3. Modélisation et Validation
- Split temporel strict : entraînement sur 2014 à septembre 2017, validation sur le dernier trimestre 2017 (octobre à décembre).
- Baselines de comparaison :
  1. Ventes de l'année précédente au même jour (Lag 364j).
  2. Moyenne mobile sur 30 jours.
- Modèle final : LightGBM Regressor avec early stopping.

## Résultats

| Approche | Type | MAE (unités) | RMSE (unités) | SMAPE (%) | WAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Moyenne Mobile (30j) | Baseline statistique | 15.83 | 19.88 | 27.99 % | 28.96 % |
| Année N-1 (Lag 364j) | Baseline métier | 8.35 | 10.90 | 17.86 % | 15.28 % |
| LightGBM Tuné | Modèle final | 5.89 | 7.66 | 12.41 % | 10.77 % |

Le modèle réduit l'erreur de prévision d'environ 30 % par rapport à la méthode métier historique (passage de 17.86 % à 12.41 % de SMAPE).

## Simulation Financière

Sur la période de validation de 3 mois (46 000 prévisions), avec des hypothèses types de gestion commerciale (15 € de marge unitaire perdue par rupture et 2 € de coût hebdomadaire par unité excédentaire) :
- Coût estimé des erreurs avec la méthode N-1 : 3 940 852 €
- Coût estimé des erreurs avec LightGBM : 2 583 480 €
- Réduction théorique des pertes logistiques : -34.4 % (~1,35 M€ d'écart sur le réseau de 10 magasins)

## Utilisation et Déploiement

### Installation

```bash
# Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Données et Exécution

1. Placer les fichiers `train.csv` et `test.csv` (Kaggle Store Item Demand Forecasting) dans `data/raw/`.
2. Exécuter les notebooks dans l'ordre :
   - `01-eda.ipynb`
   - `02-feature-engineering.ipynb`
   - `03-modelling.ipynb` (génère `data/processed/predictions_test.csv`)
3. Lancer le dashboard de restitution :

```bash
streamlit run src/app.py
```
