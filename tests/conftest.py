import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from datetime import datetime
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from src.risk_model import PedestrianRiskModel
from src.time_based_risk_model import TimeBasedPedestrianRiskModel
from src.alt_routing import ALTRouter
from src.api import app


@pytest.fixture
def sample_crash_data():
    """Fixture pour les données d'accidents de piétons"""
    data = {
        'dispatch_ts': [
            '2023-01-15 08:30:00',
            '2023-01-16 19:45:00',
            '2023-02-20 14:20:00',
            '2023-03-10 22:15:00',
            '2023-04-05 07:30:00'
        ],
        'mode_type': ['ped', 'ped', 'ped', 'ped', 'ped'],
        'lat': [42.3601, 42.3611, 42.3621, 42.3631, 42.3641],
        'long': [-71.0589, -71.0599, -71.0609, -71.0619, -71.0629]
    }
    df = pd.DataFrame(data)
    df['dispatch_ts'] = pd.to_datetime(df['dispatch_ts'])
    df['year'] = df['dispatch_ts'].dt.year
    df['month'] = df['dispatch_ts'].dt.month
    df['hour'] = df['dispatch_ts'].dt.hour
    df['day_of_week'] = df['dispatch_ts'].dt.day_name()
    return df


@pytest.fixture
def sample_311_data():
    """Fixture pour les données 311 liées aux piétons"""
    data = {
        'open_dt': [
            '2023-01-10 10:00:00',
            '2023-01-20 15:30:00',
            '2023-02-15 09:45:00',
            '2023-03-05 16:20:00'
        ],
        'case_title': [
            'Sidewalk Repair Request',
            'Street Light Out',
            'Crosswalk Maintenance',
            'Traffic Signal Issue'
        ],
        'subject': [
            'Broken sidewalk',
            'Dark street at night',
            'Faded crosswalk lines',
            'Pedestrian signal not working'
        ],
        'latitude': [42.3605, 42.3615, 42.3625, 42.3635],
        'longitude': [-71.0595, -71.0605, -71.0615, -71.0625]
    }
    df = pd.DataFrame(data)
    df['open_dt'] = pd.to_datetime(df['open_dt'])
    df['year'] = df['open_dt'].dt.year
    df['month'] = df['open_dt'].dt.month
    df['hour'] = df['open_dt'].dt.hour
    df['day_of_week'] = df['open_dt'].dt.day_name()
    df['combined_text'] = df['case_title'] + ' ' + df['subject']
    return df


@pytest.fixture
def sample_street_segments():
    """Fixture pour les segments de rue"""
    # Créer des LineString pour les segments de route
    lines = [
        LineString([(-71.0589, 42.3601), (-71.0599, 42.3611)]),
        LineString([(-71.0599, 42.3611), (-71.0609, 42.3621)]),
        LineString([(-71.0609, 42.3621), (-71.0619, 42.3631)]),
        LineString([(-71.0619, 42.3631), (-71.0629, 42.3641)])
    ]
    
    data = {
        'ST_NAME': ['Main', 'Oak', 'Pine', 'Elm'],
        'ST_TYPE': ['ST', 'AVE', 'ST', 'RD'],
        'PRE_DIR': ['', 'N', '', 'S'],
        'SUF_DIR': ['', '', 'W', ''],
        'ONEWAY': ['B', 'B', 'FT', 'B'],
        'SPEEDLIMIT': [25, 30, 35, 25],
        'CFCC': ['A41', 'A41', 'A31', 'A41'],
        'HWY_NUM': ['', '', '', '']
    }
    
    gdf = gpd.GeoDataFrame(data, geometry=lines, crs="EPSG:4326")
    
    # Créer le nom complet de la rue
    gdf['full_street_name'] = gdf.apply(
        lambda x: ' '.join(filter(None, [
            x.get('PRE_DIR', ''), 
            x.get('ST_NAME', ''), 
            x.get('ST_TYPE', ''), 
            x.get('SUF_DIR', '')
        ])),
        axis=1
    )
    
    return gdf


@pytest.fixture
def sample_crime_data():
    """Fixture pour les données de crimes"""
    data = {
        'OCCURRED_ON_DATE': [
            '2023-01-12 20:00:00',
            '2023-01-18 23:30:00',
            '2023-02-10 18:45:00',
            '2023-03-08 21:15:00'
        ],
        'OFFENSE_DESCRIPTION': [
            'ASSAULT - AGGRAVATED',
            'LARCENY THEFT - PURSE SNATCHING',
            'ROBBERY - STREET',
            'HARASSMENT ORDER VIOLATION'
        ],
        'Lat': [42.3603, 42.3613, 42.3623, 42.3633],
        'Long': [-71.0593, -71.0603, -71.0613, -71.0623]
    }
    df = pd.DataFrame(data)
    df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])
    df['year'] = df['OCCURRED_ON_DATE'].dt.year
    df['month'] = df['OCCURRED_ON_DATE'].dt.month
    df['hour'] = df['OCCURRED_ON_DATE'].dt.hour
    df['day_of_week'] = df['OCCURRED_ON_DATE'].dt.day_name()
    
    # Marquer comme lié à la sécurité des piétons
    df['ped_safety_related'] = True
    
    return df


@pytest.fixture
def sample_risk_df(sample_street_segments):
    """Fixture pour un DataFrame de risque avec des données calculées"""
    # Copier les segments de rue
    risk_df = sample_street_segments.copy()
    
    # Ajouter les colonnes de risque
    risk_df['entity_type'] = 'street_segment'
    risk_df['ped_crash_count'] = [2, 1, 3, 0]
    risk_df['ped_crash_night_count'] = [1, 0, 2, 0]
    risk_df['ped_311_count'] = [1, 2, 1, 0]
    risk_df['sidewalk_issues'] = [1, 0, 0, 0]
    risk_df['crossing_issues'] = [0, 1, 1, 0]
    risk_df['lighting_issues'] = [0, 1, 0, 0]
    risk_df['signal_issues'] = [0, 0, 1, 0]
    risk_df['crime_count'] = [1, 0, 2, 0]
    risk_df['buffer_distance'] = 50
    risk_df['risk_score'] = [0.8, 0.4, 0.9, 0.1]
    risk_df['risk_level'] = ['Élevé', 'Modéré', 'Très élevé', 'Faible']
    risk_df['cluster'] = [0, -1, 0, -1]
    
    return risk_df


@pytest.fixture
def mock_pedestrian_risk_model(sample_risk_df, sample_street_segments):
    """Fixture pour un modèle de risque mocké"""
    model = Mock(spec=PedestrianRiskModel)
    model.risk_df = sample_risk_df
    model.streets_gdf = sample_street_segments
    model.intersections_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return model


@pytest.fixture
def mock_time_model():
    """Fixture pour un modèle temporel mocké"""
    model = Mock(spec=TimeBasedPedestrianRiskModel)
    model.hourly_risk_scores = {hour: np.random.random() for hour in range(24)}
    model.daily_risk_scores = {
        'Monday': 0.6, 'Tuesday': 0.5, 'Wednesday': 0.5,
        'Thursday': 0.6, 'Friday': 0.8, 'Saturday': 0.9, 'Sunday': 0.7
    }
    model.monthly_risk_scores = {month: np.random.random() for month in range(1, 13)}
    return model


@pytest.fixture
def sample_alt_router(mock_pedestrian_risk_model):
    """Fixture pour un routeur ALT"""
    router = ALTRouter(mock_pedestrian_risk_model)
    return router


@pytest.fixture
def flask_app():
    """Fixture pour l'application Flask"""
    app.config['TESTING'] = True
    return app


@pytest.fixture
def flask_client(flask_app):
    """Fixture pour le client de test Flask"""
    return flask_app.test_client()


@pytest.fixture
def temp_data_dir():
    """Fixture pour un répertoire temporaire avec des fichiers de données"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Créer des fichiers CSV fictifs
        crash_data = {
            'dispatch_ts': ['2023-01-15T08:30:00', '2023-01-16T19:45:00'],
            'mode_type': ['ped', 'ped'],
            'lat': [42.3601, 42.3611],
            'long': [-71.0589, -71.0599]
        }
        pd.DataFrame(crash_data).to_csv(temp_path / 'vision-zero-crash-records.csv', index=False)
        
        # Fichier 311
        data_311 = {
            'open_dt': ['2023-01-10T10:00:00', '2023-01-20T15:30:00'],
            'case_title': ['Sidewalk Repair', 'Street Light'],
            'subject': ['Broken sidewalk', 'Dark street'],
            'latitude': [42.3605, 42.3615],
            'longitude': [-71.0595, -71.0605]
        }
        pd.DataFrame(data_311).to_csv(temp_path / '311.csv', index=False)
        
        # Fichier crimes
        crime_data = {
            'OCCURRED_ON_DATE': ['2023-01-12T20:00:00', '2023-01-18T23:30:00'],
            'OFFENSE_DESCRIPTION': ['ASSAULT', 'LARCENY'],
            'Lat': [42.3603, 42.3613],
            'Long': [-71.0593, -71.0603]
        }
        pd.DataFrame(crime_data).to_csv(temp_path / 'crimes-incident-report.csv', index=False)
        
        # Fichier GeoJSON pour les segments de rue
        lines = [
            LineString([(-71.0589, 42.3601), (-71.0599, 42.3611)]),
            LineString([(-71.0599, 42.3611), (-71.0609, 42.3621)])
        ]
        
        street_data = {
            'ST_NAME': ['Main', 'Oak'],
            'ST_TYPE': ['ST', 'AVE'],
            'PRE_DIR': ['', 'N'],
            'SUF_DIR': ['', ''],
            'ONEWAY': ['B', 'B'],
            'SPEEDLIMIT': [25, 30]
        }
        
        gdf_streets = gpd.GeoDataFrame(street_data, geometry=lines, crs="EPSG:4326")
        gdf_streets.to_file(temp_path / 'boston-street-segments.geojson', driver='GeoJSON')
        
        yield temp_path


@pytest.fixture
def mock_datetime_now():
    """Fixture pour mocker datetime.now()"""
    test_datetime = datetime(2023, 6, 15, 14, 30, 0)  # Jeudi 15 juin 2023, 14h30
    with patch('datetime.datetime') as mock_dt:
        mock_dt.now.return_value = test_datetime
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield test_datetime


@pytest.fixture
def sample_coordinates():
    """Fixture pour des coordonnées d'exemple"""
    return {
        'start': {'lat': 42.3601, 'lon': -71.0589},
        'end': {'lat': 42.3641, 'lon': -71.0629}
    }


class MockResponse:
    """Classe pour mocker les réponses HTTP"""
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data


@pytest.fixture
def mock_requests():
    """Fixture pour mocker les requêtes HTTP"""
    def _mock_requests(url, **kwargs):
        if 'overpass' in url:
            return MockResponse({'elements': []}, 200)
        return MockResponse({}, 404)
    
    with patch('requests.get', side_effect=_mock_requests):
        yield


# Fixtures pour les tests d'intégration
@pytest.fixture
def integrated_model(temp_data_dir):
    """Fixture pour un modèle intégré avec données temporaires"""
    with patch('pathlib.Path.__truediv__', side_effect=lambda self, other: temp_data_dir / other if 'data' in str(self) else self / other):
        model = PedestrianRiskModel()
        yield model


@pytest.fixture
def sample_geojson_feature():
    """Fixture pour une feature GeoJSON"""
    return {
        "type": "Feature",
        "properties": {
            "name": "Test Street",
            "risk_score": 0.75,
            "risk_level": "Élevé"
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [[-71.0589, 42.3601], [-71.0599, 42.3611]]
        }
    }


@pytest.fixture
def sample_cluster_data():
    """Fixture pour des données de cluster"""
    return [
        {
            'cluster_id': 0,
            'size': 3,
            'type': 'intersection',
            'avg_risk_score': 0.75,
            'max_risk_score': 0.9,
            'total_pedestrian_crashes': 5,
            'coordinates': {'lat': 42.3601, 'lon': -71.0589}
        },
        {
            'cluster_id': 1,
            'size': 2,
            'type': 'street_segment',
            'avg_risk_score': 0.6,
            'max_risk_score': 0.7,
            'total_pedestrian_crashes': 3,
            'coordinates': {'lat': 42.3611, 'lon': -71.0599}
        }
    ]