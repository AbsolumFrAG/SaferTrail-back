# SaferTrail Boston 🚶‍♀️🚦

**Modèle d'analyse de la sécurité piétonne à Boston avec routage intelligent**

SaferTrail est un système d'analyse avancé qui évalue les risques pour les piétons à Boston en combinant les données d'accidents, les signalements 311, les incidents criminels et les caractéristiques des infrastructures. Le système propose également un routage intelligent qui évite les zones dangereuses et tient compte des restrictions piétonnes.

## 🌟 Fonctionnalités principales

- **Analyse de risque spatiale** : Identification des zones dangereuses pour les piétons
- **Prédiction temporelle** : Évaluation du risque selon l'heure, le jour et la saison
- **Routage sécurisé ALT** : Algorithme de routage évitant les zones à risque avec détection des transports requis
- **Classification des routes** : Filtrage automatique des routes impraticables à pied
- **API REST complète** : Interface pour intégration dans d'autres applications
- **Clustering de zones à risque** : Identification des points chauds de danger
- **Recommandations de sécurité** : Suggestions d'amélioration pour chaque zone

## 📋 Prérequis

- Python 3.8+
- pip
- Données géospatiales de Boston (voir section Structure des données)

## 🚀 Installation et configuration

### 1. Cloner le projet

```bash
git clone <votre-repo>
cd safertrail-boston
```

### 2. Créer l'environnement virtuel

```bash
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur macOS/Linux :
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Structure des données

Créez un dossier `data/` à la racine du projet et ajoutez les fichiers suivants :

```
data/
├── vision-zero-crash-records.csv      # Données d'accidents Vision Zero
├── 311.csv                            # Signalements 311
├── boston-street-segments.geojson     # Segments de rue de Boston
└── crimes-incident-report.csv         # Rapports de crimes
```

**Sources de données recommandées :**
- [Boston Vision Zero Crash Records](https://data.boston.gov/)
- [Boston 311 Service Requests](https://data.boston.gov/)
- [Boston Street Segments](https://bostonopendata-boston.opendata.arcgis.com/)
- [Boston Crime Incident Reports](https://data.boston.gov/)

### 5. Créer les dossiers de modèles

```bash
mkdir models
```

## 🎯 Utilisation

### Interface en ligne de commande (CLI)

#### Entraîner le modèle

```bash
# Premier entraînement
python main.py train

# Forcer le réentraînement
python main.py train --force
```

#### Analyser les zones à risque

```bash
# Obtenir les 10 zones les plus dangereuses (seuil 0.7)
python main.py risk-zones

# Personnaliser les paramètres
python main.py risk-zones --threshold 0.5 --limit 20 --output zones_risque.json
```

#### Identifier les clusters de danger

```bash
# Clusters avec minimum 2 zones
python main.py clusters

# Clusters plus grands
python main.py clusters --min-size 5 --output clusters_danger.json
```

#### Obtenir des recommandations

```bash
# Recommandations pour une zone spécifique
python main.py recommendations 123

# Sauvegarder les recommandations
python main.py recommendations 123 --output recommandations_zone_123.json
```

#### Générer des statistiques

```bash
# Statistiques globales
python main.py statistics --output stats_globales.json
```

### API REST

#### Démarrer l'API

```bash
# Démarrage simple
python main.py api

# Configuration personnalisée
python main.py api --host 0.0.0.0 --port 8080 --debug
```

L'API sera disponible à `http://localhost:8000` avec la documentation Swagger à `http://localhost:8000/docs`

#### Endpoints principaux

**Analyse de risque :**
- `GET /risk-zones` - Zones à haut risque
- `GET /clusters` - Clusters de zones dangereuses
- `GET /recommendations/<zone_id>` - Recommandations pour une zone
- `GET /statistics` - Statistiques globales

**Analyse temporelle :**
- `GET /current-risk` - Risque actuel
- `GET /time-risk?hour=20&day=Friday` - Risque à un moment spécifique
- `GET /risk-forecast?hours=24` - Prévision sur 24h
- `GET /high-risk-times` - Périodes les plus dangereuses

**Routage sécurisé :**
- `GET /safe-route` - Itinéraire sécurisé
- `GET /compare-routes` - Comparaison itinéraire sûr vs court
- `GET /routing-settings` - Configuration du routeur

**Exemples d'utilisation :**

```bash
# Risque actuel
curl http://localhost:8000/current-risk

# Itinéraire sécurisé
curl "http://localhost:8000/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3584&end_lon=-71.0598"

# Zones à risque en GeoJSON
curl "http://localhost:8000/risk-zones?format=geojson&threshold=0.6"
```

## 🏗️ Architecture du projet

```
safertrail-boston/
├── main.py                    # Point d'entrée CLI
├── requirements.txt           # Dépendances Python
├── README.md                 # Documentation
├── data/                     # Données sources (à créer)
├── models/                   # Modèles entraînés (à créer)
└── src/
    ├── risk_model.py         # Modèle principal d'analyse des risques
    ├── time_based_risk_model.py  # Modèle de prédiction temporelle
    ├── alt_routing.py        # Algorithme de routage ALT
    ├── api.py               # API REST Flask
    ├── data_loader.py       # Chargement des données
    ├── data_processor.py    # Traitement et analyse des données
    └── geo_utils.py         # Utilitaires géospatiaux
```

## 🧭 Fonctionnalités du routage

### Classification automatique des routes

Le système classe automatiquement les routes selon leur praticabilité pour les piétons :

- **Routes interdites** : Autoroutes, voies rapides (> 55 mph)
- **Transports requis** : Ferries, certains ponts/tunnels
- **Routes avec avertissements** : Zones industrielles, ruelles, routes à trafic dense

### Algorithme ALT (A* avec Landmarks)

- Optimisation par landmarks pour un routage rapide
- Prise en compte du risque piéton et de l'heure
- Évitement automatique des routes dangereuses
- Support des transports en commun et ferries

### Paramètres configurables

```bash
# Configurer l'importance du facteur risque vs distance
curl -X POST http://localhost:8000/routing-settings \
  -H "Content-Type: application/json" \
  -d '{"risk_weight": 0.8, "max_detour_factor": 1.5}'
```

## 📊 Exemple de workflow complet

```bash
# 1. Entraîner le modèle
python main.py train

# 2. Analyser les zones dangereuses
python main.py risk-zones --threshold 0.6 --output zones_danger.json

# 3. Identifier les clusters
python main.py clusters --min-size 3 --output clusters.json

# 4. Démarrer l'API pour l'utilisation en temps réel
python main.py api --port 8000

# 5. Tester le routage sécurisé
curl "http://localhost:8000/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3584&end_lon=-71.0598&format=geojson"
```

## 🔧 Développement

### Ajouter de nouvelles fonctionnalités

1. **Nouveaux facteurs de risque** : Modifier `data_processor.py`
2. **Algorithmes de routage** : Étendre `alt_routing.py`
3. **Endpoints API** : Ajouter dans `api.py`
4. **Analyses temporelles** : Développer `time_based_risk_model.py`

### Tests

```bash
# Lancer les tests (si implémentés)
python -m pytest tests/
```

## 📈 Performance

- **Entraînement** : ~2-5 minutes selon la taille des données
- **Routage** : <1 seconde pour des distances moyennes
- **API** : Support de plusieurs requêtes simultanées
- **Mémoire** : ~500MB-2GB selon la densité du réseau routier

## ⚠️ Limitations

- Données limitées à Boston
- Qualité dépendante des données sources
- Algorithme de routage optimisé pour piétons uniquement
- Prédictions basées sur données historiques

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les modifications (`git commit -am 'Ajouter nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🆘 Support

- **Issues** : Utiliser le système d'issues GitHub
- **Documentation API** : `http://localhost:8000/docs` après démarrage
- **Logs** : Les erreurs sont affichées dans la console

## 🙏 Remerciements

- Ville de Boston pour les données ouvertes
- Communauté Vision Zero
- Contributeurs des bibliothèques open source utilisées

---

**Fait avec ❤️ pour la sécurité des piétons à Boston**