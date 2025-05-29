# SaferTrail Boston 🚶‍♀️🚦

**Pedestrian safety analysis model for Boston with intelligent routing**

SaferTrail is an advanced analysis system that evaluates pedestrian risks in Boston by combining crash data, 311 reports, criminal incidents, and infrastructure characteristics. The system also offers intelligent routing that avoids dangerous areas and takes pedestrian restrictions into account.

## 🌟 Main Features

- **Spatial risk analysis**: Identification of dangerous zones for pedestrians
- **Temporal prediction**: Risk assessment based on time, day, and season
- **Secure ALT routing**: Routing algorithm that avoids risk zones with required transport detection
- **Road classification**: Automatic filtering of roads impassable on foot
- **Complete REST API**: Interface for integration into other applications
- **Risk zone clustering**: Identification of danger hotspots
- **Safety recommendations**: Improvement suggestions for each zone

## 📋 Prerequisites

- Python 3.8+
- pip
- Boston geospatial data (see Data Structure section)

## 🚀 Installation and Setup

### 1. Clone the project

```bash
git clone <your-repo>
cd safertrail-boston
```

### 2. Create virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Data structure

Create a `data/` folder at the project root and add the following files:

```
data/
├── vision-zero-crash-records.csv      # Vision Zero crash data
├── 311.csv                            # 311 reports
├── boston-street-segments.geojson     # Boston street segments
└── crimes-incident-report.csv         # Crime reports
```

**Recommended data sources:**
- [Boston Vision Zero Crash Records](https://data.boston.gov/)
- [Boston 311 Service Requests](https://data.boston.gov/)
- [Boston Street Segments](https://bostonopendata-boston.opendata.arcgis.com/)
- [Boston Crime Incident Reports](https://data.boston.gov/)

### 5. Create model directories

```bash
mkdir models
```

## 🎯 Usage

### Command Line Interface (CLI)

#### Train the model

```bash
# Initial training
python main.py train

# Force retraining
python main.py train --force
```

#### Analyze risk zones

```bash
# Get the 10 most dangerous zones (threshold 0.7)
python main.py risk-zones

# Customize parameters
python main.py risk-zones --threshold 0.5 --limit 20 --output risk_zones.json
```

#### Identify danger clusters

```bash
# Clusters with minimum 2 zones
python main.py clusters

# Larger clusters
python main.py clusters --min-size 5 --output danger_clusters.json
```

#### Get recommendations

```bash
# Recommendations for a specific zone
python main.py recommendations 123

# Save recommendations
python main.py recommendations 123 --output zone_123_recommendations.json
```

#### Generate statistics

```bash
# Global statistics
python main.py statistics --output global_stats.json
```

### REST API

#### Start the API

```bash
# Simple startup
python main.py api

# Custom configuration
python main.py api --host 0.0.0.0 --port 8080 --debug
```

The API will be available at `http://localhost:8000` with Swagger documentation at `http://localhost:8000/docs`

#### Main endpoints

**Risk analysis:**
- `GET /risk-zones` - High-risk zones
- `GET /clusters` - Dangerous zone clusters
- `GET /recommendations/<zone_id>` - Recommendations for a zone
- `GET /statistics` - Global statistics

**Temporal analysis:**
- `GET /current-risk` - Current risk
- `GET /time-risk?hour=20&day=Friday` - Risk at a specific time
- `GET /risk-forecast?hours=24` - 24-hour forecast
- `GET /high-risk-times` - Most dangerous periods

**Secure routing:**
- `GET /safe-route` - Secure route
- `GET /compare-routes` - Safe vs short route comparison
- `GET /routing-settings` - Router configuration

**Usage examples:**

```bash
# Current risk
curl http://localhost:8000/current-risk

# Secure route
curl "http://localhost:8000/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3584&end_lon=-71.0598"

# Risk zones in GeoJSON
curl "http://localhost:8000/risk-zones?format=geojson&threshold=0.6"
```

## 🏗️ Project Architecture

```
safertrail-boston/
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── README.md                 # Documentation
├── data/                     # Source data (to create)
├── models/                   # Trained models (to create)
└── src/
    ├── risk_model.py         # Main risk analysis model
    ├── time_based_risk_model.py  # Temporal prediction model
    ├── alt_routing.py        # ALT routing algorithm
    ├── api.py               # Flask REST API
    ├── data_loader.py       # Data loading
    ├── data_processor.py    # Data processing and analysis
    └── geo_utils.py         # Geospatial utilities
```

## 🧭 Routing Features

### Automatic road classification

The system automatically classifies roads according to their walkability for pedestrians:

- **Prohibited roads**: Highways, expressways (> 55 mph)
- **Transport required**: Ferries, certain bridges/tunnels
- **Roads with warnings**: Industrial areas, alleys, high-traffic roads

### ALT Algorithm (A* with Landmarks)

- Landmark optimization for fast routing
- Considers pedestrian risk and time
- Automatic avoidance of dangerous roads
- Support for public transport and ferries

### Configurable parameters

```bash
# Configure risk factor importance vs distance
curl -X POST http://localhost:8000/routing-settings \
  -H "Content-Type: application/json" \
  -d '{"risk_weight": 0.8, "max_detour_factor": 1.5}'
```

## 📊 Complete Workflow Example

```bash
# 1. Train the model
python main.py train

# 2. Analyze dangerous zones
python main.py risk-zones --threshold 0.6 --output danger_zones.json

# 3. Identify clusters
python main.py clusters --min-size 3 --output clusters.json

# 4. Start API for real-time use
python main.py api --port 8000

# 5. Test secure routing
curl "http://localhost:8000/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3584&end_lon=-71.0598&format=geojson"
```

## 🔧 Development

### Adding new features

1. **New risk factors**: Modify `data_processor.py`
2. **Routing algorithms**: Extend `alt_routing.py`
3. **API endpoints**: Add to `api.py`
4. **Temporal analysis**: Develop `time_based_risk_model.py`

### Testing

```bash
# Run tests (if implemented)
python -m pytest tests/
```

## 📈 Performance

- **Training**: ~2-5 minutes depending on data size
- **Routing**: <1 second for average distances
- **API**: Support for multiple simultaneous requests
- **Memory**: ~500MB-2GB depending on road network density

## ⚠️ Limitations

- Data limited to Boston
- Quality dependent on source data
- Routing algorithm optimized for pedestrians only
- Predictions based on historical data

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is under MIT license. See the `LICENSE` file for more details.

## 🆘 Support

- **Issues**: Use GitHub issue system
- **API Documentation**: `http://localhost:8000/docs` after startup
- **Logs**: Errors are displayed in console

## 🙏 Acknowledgments

- City of Boston for open data
- Vision Zero community
- Contributors to the open source libraries used

---

**Made with ❤️ for pedestrian safety in Boston**