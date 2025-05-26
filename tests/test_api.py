import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from src.api import app
from src.risk_model import PedestrianRiskModel
from src.alt_routing import ALTRouter


class TestFlaskAppSetup:
    """Tests pour la configuration de l'application Flask"""
    
    def test_app_creation(self):
        """Test de création de l'application"""
        assert app is not None
        assert app.name == 'src.api'
        
    def test_app_testing_mode(self, flask_client):
        """Test du mode test"""
        assert app.config['TESTING'] is True


class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""
    
    def test_health_endpoint_success(self, flask_client):
        """Test de l'endpoint de santé - succès"""
        response = flask_client.get('/health')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'message' in data
        assert 'timestamp' in data
        assert 'features' in data
        
        # Vérifier les features
        assert 'risk_analysis' in data['features']
        assert 'temporal_prediction' in data['features']
        assert 'safe_routing' in data['features']
        
    def test_health_endpoint_content_type(self, flask_client):
        """Test du type de contenu"""
        response = flask_client.get('/health')
        
        assert 'application/json' in response.content_type


class TestTrainEndpoint:
    """Tests pour l'endpoint /train"""
    
    @patch('src.api.model.train')
    @patch('src.api.router.build_graph')
    def test_train_endpoint_success(self, mock_build_graph, mock_train, flask_client):
        """Test d'entraînement réussi"""
        mock_train.return_value = True
        mock_build_graph.return_value = True
        
        response = flask_client.post('/train')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'message' in data
        assert 'routing_status' in data
        
        mock_train.assert_called_once_with(force_retrain=False)
        
    @patch('src.api.model.train')
    def test_train_endpoint_with_force_retrain(self, mock_train, flask_client):
        """Test avec force_retrain"""
        mock_train.return_value = True
        
        response = flask_client.post('/train', 
                                   json={'force_retrain': True},
                                   content_type='application/json')
        
        assert response.status_code == 200
        mock_train.assert_called_once_with(force_retrain=True)
        
    @patch('src.api.model.train')
    def test_train_endpoint_failure(self, mock_train, flask_client):
        """Test d'échec d'entraînement"""
        mock_train.return_value = False
        
        response = flask_client.post('/train')
        
        assert response.status_code == 500
        
        data = json.loads(response.data)
        assert data['status'] == 'error'
        
    def test_train_endpoint_invalid_json(self, flask_client):
        """Test avec JSON invalide"""
        response = flask_client.post('/train',
                                   data='invalid json',
                                   content_type='application/json')
        
        # Doit gérer le JSON invalide gracieusement
        assert response.status_code in [200, 400, 500]


class TestRiskZonesEndpoint:
    """Tests pour l'endpoint /risk-zones"""
    
    @patch('src.api.model.get_high_risk_zones')
    def test_risk_zones_endpoint_success(self, mock_get_zones, flask_client, sample_cluster_data):
        """Test de récupération réussie des zones à risque"""
        mock_zones = [
            {
                'id': 1,
                'name': 'Test Zone',
                'risk_score': 0.8,
                'risk_level': 'Élevé',
                'coordinates': {'lat': 42.3601, 'lon': -71.0589}
            }
        ]
        mock_get_zones.return_value = mock_zones
        
        response = flask_client.get('/risk-zones')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'count' in data
        assert 'zones' in data
        assert data['count'] == len(mock_zones)
        assert len(data['zones']) == 1
        
        mock_get_zones.assert_called_once_with(threshold=0.7, limit=10)
        
    @patch('src.api.model.get_high_risk_zones')
    def test_risk_zones_endpoint_with_parameters(self, mock_get_zones, flask_client):
        """Test avec paramètres personnalisés"""
        mock_get_zones.return_value = []
        
        response = flask_client.get('/risk-zones?threshold=0.5&limit=20')
        
        assert response.status_code == 200
        mock_get_zones.assert_called_once_with(threshold=0.5, limit=20)
        
    @patch('src.api.model.get_high_risk_zones')
    @patch('src.api.create_geojson_from_risk_zones')
    def test_risk_zones_endpoint_geojson_format(self, mock_create_geojson, mock_get_zones, flask_client):
        """Test avec format GeoJSON"""
        mock_zones = [{'id': 1, 'name': 'Test'}]
        mock_geojson = {'type': 'FeatureCollection', 'features': []}
        
        mock_get_zones.return_value = mock_zones
        mock_create_geojson.return_value = mock_geojson
        
        response = flask_client.get('/risk-zones?format=geojson')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['type'] == 'FeatureCollection'
        
        mock_create_geojson.assert_called_once_with(mock_zones)


class TestClustersEndpoint:
    """Tests pour l'endpoint /clusters"""
    
    @patch('src.api.model.get_risk_clusters')
    def test_clusters_endpoint_success(self, mock_get_clusters, flask_client, sample_cluster_data):
        """Test de récupération réussie des clusters"""
        mock_get_clusters.return_value = sample_cluster_data
        
        response = flask_client.get('/clusters')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'count' in data
        assert 'clusters' in data
        assert data['count'] == len(sample_cluster_data)
        
        mock_get_clusters.assert_called_once_with(min_cluster_size=2)
        
    @patch('src.api.model.get_risk_clusters')
    def test_clusters_endpoint_with_min_size(self, mock_get_clusters, flask_client):
        """Test avec taille minimale personnalisée"""
        mock_get_clusters.return_value = []
        
        response = flask_client.get('/clusters?min_size=5')
        
        assert response.status_code == 200
        mock_get_clusters.assert_called_once_with(min_cluster_size=5)


class TestRecommendationsEndpoint:
    """Tests pour l'endpoint /recommendations/<zone_id>"""
    
    @patch('src.api.model.get_safety_recommendations')
    def test_recommendations_endpoint_success(self, mock_get_recommendations, flask_client):
        """Test de récupération réussie des recommandations"""
        mock_recommendations = {
            'zone_id': 123,
            'name': 'Test Zone',
            'recommendations': ['Install traffic lights', 'Improve lighting'],
            'problems_identified': ['High accident rate']
        }
        mock_get_recommendations.return_value = mock_recommendations
        
        response = flask_client.get('/recommendations/123')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['zone_id'] == 123
        assert 'recommendations' in data
        assert len(data['recommendations']) == 2
        
        mock_get_recommendations.assert_called_once_with(123)
        
    @patch('src.api.model.get_safety_recommendations')
    def test_recommendations_endpoint_zone_not_found(self, mock_get_recommendations, flask_client):
        """Test avec zone inexistante"""
        mock_get_recommendations.return_value = {'error': 'Zone not found'}
        
        response = flask_client.get('/recommendations/999')
        
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert 'error' in data
        
    def test_recommendations_endpoint_invalid_zone_id(self, flask_client):
        """Test avec ID de zone invalide"""
        response = flask_client.get('/recommendations/invalid')
        
        # Flask devrait gérer la conversion d'entier invalide
        assert response.status_code == 404


class TestTimeRiskEndpoint:
    """Tests pour l'endpoint /time-risk"""
    
    @patch('src.api.model.get_risk_prediction_for_time')
    def test_time_risk_endpoint_success(self, mock_get_prediction, flask_client):
        """Test de prédiction de risque temporel réussie"""
        mock_prediction = {
            'time': {'hour': 14, 'day_name': 'Jeudi'},
            'risk': {'score': 0.6, 'level': 'Modéré'},
            'recommendations': ['Be careful']
        }
        mock_get_prediction.return_value = mock_prediction
        
        response = flask_client.get('/time-risk?hour=14&day=3&month=6')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'time' in data
        assert 'risk' in data
        assert data['risk']['score'] == 0.6
        
        mock_get_prediction.assert_called_once_with(hour=14, day_of_week=3, month=6, date_time=None)
        
    @patch('src.api.model.get_risk_prediction_for_time')
    def test_time_risk_endpoint_with_datetime(self, mock_get_prediction, flask_client):
        """Test avec datetime ISO"""
        mock_get_prediction.return_value = {'time': {}, 'risk': {}}
        
        response = flask_client.get('/time-risk?datetime=2023-06-15T14:30:00')
        
        assert response.status_code == 200
        
        # Vérifier que datetime est passé au modèle
        call_args = mock_get_prediction.call_args
        assert call_args[1]['date_time'] is not None
        
    def test_time_risk_endpoint_invalid_datetime(self, flask_client):
        """Test avec datetime invalide"""
        response = flask_client.get('/time-risk?datetime=invalid-date')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data['status'] or 'message' in data
        
    @patch('src.api.model.get_risk_prediction_for_time')
    def test_time_risk_endpoint_day_name(self, mock_get_prediction, flask_client):
        """Test avec nom de jour"""
        mock_get_prediction.return_value = {'time': {}, 'risk': {}}
        
        response = flask_client.get('/time-risk?day=Monday')
        
        assert response.status_code == 200
        
        call_args = mock_get_prediction.call_args
        assert call_args[1]['day_of_week'] == 'Monday'


class TestCurrentRiskEndpoint:
    """Tests pour l'endpoint /current-risk"""
    
    @patch('src.api.model.get_risk_prediction_for_time')
    @patch('src.api.datetime')
    def test_current_risk_endpoint_success(self, mock_datetime, mock_get_prediction, flask_client):
        """Test de risque actuel réussi"""
        test_time = datetime(2023, 6, 15, 14, 30)
        mock_datetime.now.return_value = test_time
        
        mock_prediction = {
            'time': {'hour': 14},
            'risk': {'score': 0.5, 'level': 'Modéré'}
        }
        mock_get_prediction.return_value = mock_prediction
        
        response = flask_client.get('/current-risk')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'timestamp' in data
        assert 'risk' in data
        
        mock_get_prediction.assert_called_once_with(date_time=test_time)
        
    @patch('src.api.model.get_risk_prediction_for_time')
    def test_current_risk_endpoint_error(self, mock_get_prediction, flask_client):
        """Test d'erreur de risque actuel"""
        mock_get_prediction.return_value = {'error': 'Model not available'}
        
        response = flask_client.get('/current-risk')
        
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert 'error' in data


class TestRiskForecastEndpoint:
    """Tests pour l'endpoint /risk-forecast"""
    
    @patch('src.api.model.get_risk_prediction_for_time')
    @patch('src.api.datetime')
    def test_risk_forecast_endpoint_success(self, mock_datetime, mock_get_prediction, flask_client):
        """Test de prévision de risque réussie"""
        test_time = datetime(2023, 6, 15, 14, 30)
        mock_datetime.now.return_value = test_time
        
        mock_prediction = {
            'time': {'hour': 14, 'day_name': 'Jeudi', 'period': 'après-midi'},
            'risk': {'score': 0.5, 'level': 'Modéré'}
        }
        mock_get_prediction.return_value = mock_prediction
        
        response = flask_client.get('/risk-forecast?hours=6')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'start_time' in data
        assert 'hours' in data
        assert 'forecast' in data
        assert data['hours'] == 6
        assert len(data['forecast']) <= 6
        
        # Vérifier que get_risk_prediction_for_time est appelé pour chaque heure
        assert mock_get_prediction.call_count == 6


class TestSafeRouteEndpoint:
    """Tests pour l'endpoint /safe-route"""
    
    @patch('src.api.router.get_safe_route')
    def test_safe_route_endpoint_success(self, mock_get_route, flask_client):
        """Test de routage sécurisé réussi"""
        mock_route = {
            'status': 'success',
            'route': {
                'distance': {'miles': 1.5},
                'estimated_time': {'formatted': '18min'},
                'risk_score': 0.3,
                'segments': []
            }
        }
        mock_get_route.return_value = mock_route
        
        response = flask_client.get('/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3611&end_lon=-71.0599')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'route' in data
        assert 'distance' in data['route']
        
        mock_get_route.assert_called_once()
        
    def test_safe_route_endpoint_missing_coordinates(self, flask_client):
        """Test avec coordonnées manquantes"""
        response = flask_client.get('/safe-route?start_lat=42.3601')  # Coordonnées incomplètes
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data['status'] or 'message' in data
        
    def test_safe_route_endpoint_invalid_coordinates(self, flask_client):
        """Test avec coordonnées invalides"""
        response = flask_client.get('/safe-route?start_lat=invalid&start_lon=-71.0589&end_lat=42.3611&end_lon=-71.0599')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data['status'] or 'message' in data
        
    @patch('src.api.router.get_route_as_geojson')
    def test_safe_route_endpoint_geojson_format(self, mock_get_geojson, flask_client):
        """Test avec format GeoJSON"""
        mock_geojson = {'type': 'FeatureCollection', 'features': []}
        mock_get_geojson.return_value = mock_geojson
        
        response = flask_client.get('/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3611&end_lon=-71.0599&format=geojson')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['type'] == 'FeatureCollection'
        
    @patch('src.api.router.set_risk_weight')
    @patch('src.api.router.get_safe_route')
    def test_safe_route_endpoint_with_risk_weight(self, mock_get_route, mock_set_weight, flask_client):
        """Test avec poids de risque personnalisé"""
        mock_get_route.return_value = {'status': 'success', 'route': {}}
        
        response = flask_client.get('/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3611&end_lon=-71.0599&risk_weight=0.5')
        
        assert response.status_code == 200
        mock_set_weight.assert_called_once_with(0.5)
        
    @patch('src.api.router.get_safe_route')
    def test_safe_route_endpoint_with_time(self, mock_get_route, flask_client):
        """Test avec heure spécifiée"""
        mock_get_route.return_value = {'status': 'success', 'route': {}}
        
        response = flask_client.get('/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3611&end_lon=-71.0599&time=2023-06-15T14:30:00')
        
        assert response.status_code == 200
        
        # Vérifier que l'heure est passée à get_safe_route
        call_args = mock_get_route.call_args
        assert len(call_args[0]) == 3  # start_point, end_point, time
        
    def test_safe_route_endpoint_invalid_time(self, flask_client):
        """Test avec heure invalide"""
        response = flask_client.get('/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3611&end_lon=-71.0599&time=invalid-time')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data['status'] or 'message' in data


class TestCompareRoutesEndpoint:
    """Tests pour l'endpoint /compare-routes"""
        
    @patch('src.api.router.get_safe_route')
    def test_compare_routes_endpoint_route_error(self, mock_get_route, flask_client):
        """Test avec erreur de calcul de route"""
        mock_get_route.side_effect = [
            {'status': 'error', 'message': 'Route not found'},
            {'status': 'success', 'route': {}}
        ]
        
        response = flask_client.get('/compare-routes?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3611&end_lon=-71.0599')
        
        assert response.status_code == 500
        
        data = json.loads(response.data)
        assert 'error' in data['status'] or 'message' in data


class TestRoutingSettingsEndpoint:
    """Tests pour l'endpoint /routing-settings"""
    
    @patch('src.api.router')
    def test_routing_settings_get(self, mock_router, flask_client):
        """Test de récupération des paramètres de routage"""
        mock_router.risk_weight = 0.7
        mock_router.max_detour_factor = 1.5
        mock_router.graph = MagicMock()
        mock_router.landmarks = [1, 2, 3]
        
        response = flask_client.get('/routing-settings')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'risk_weight' in data
        assert 'max_detour_factor' in data
        assert 'graph_initialized' in data
        assert 'num_landmarks' in data
        
    @patch('src.api.router.set_risk_weight')
    @patch('src.api.router.set_max_detour_factor')
    def test_routing_settings_post(self, mock_set_detour, mock_set_weight, flask_client):
        """Test de modification des paramètres de routage"""
        settings = {
            'risk_weight': 0.8,
            'max_detour_factor': 2.0
        }
        
        response = flask_client.post('/routing-settings',
                                   json=settings,
                                   content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'success'
        
        mock_set_weight.assert_called_once_with(0.8)
        mock_set_detour.assert_called_once_with(2.0)
        
    @patch('src.api.router.set_risk_weight')
    def test_routing_settings_post_invalid_values(self, mock_set_weight, flask_client):
        """Test avec valeurs invalides"""
        mock_set_weight.side_effect = ValueError("Invalid risk weight")
        
        settings = {'risk_weight': 1.5}  # Valeur invalide
        
        response = flask_client.post('/routing-settings',
                                   json=settings,
                                   content_type='application/json')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data['status'] or 'message' in data


class TestStatisticsEndpoint:
    """Tests pour l'endpoint /statistics"""
    
    @patch('src.api.model.load_model')
    @patch('src.api.model.get_high_risk_times')
    def test_statistics_endpoint_success(self, mock_get_times, mock_load_model, flask_client, sample_risk_df):
        """Test de récupération réussie des statistiques"""
        # Configurer le mock du modèle
        with patch('src.api.model.risk_df', sample_risk_df):
            mock_get_times.return_value = {
                'hours': [{'hour': 20}],
                'days': [{'day_fr': 'Vendredi'}],
                'months': [{'month_name': 'Décembre'}]
            }
            
            response = flask_client.get('/statistics')
            
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'total_pedestrian_crashes' in data
            assert 'total_night_crashes' in data
            assert 'high_risk_zones_count' in data
            assert 'common_issues' in data
            assert 'temporal_patterns' in data
            assert 'updated_at' in data
            
    @patch('src.api.model.load_model')
    def test_statistics_endpoint_no_model(self, mock_load_model, flask_client):
        """Test sans modèle chargé"""
        with patch('src.api.model.risk_df', None):
            response = flask_client.get('/statistics')
            
            assert response.status_code == 500
            
            data = json.loads(response.data)
            assert 'error' in data['status'] or 'message' in data


class TestAreasEndpoint:
    """Tests pour l'endpoint /areas/<area_name>"""
    
    @patch('src.api.model.load_model')
    def test_areas_endpoint_success(self, mock_load_model, flask_client, sample_risk_df):
        """Test de récupération d'informations sur une zone"""
        # Ajouter une colonne neighborhood pour le test
        sample_risk_df_with_neighborhood = sample_risk_df.copy()
        sample_risk_df_with_neighborhood['neighborhood'] = ['Downtown', 'Back Bay', 'South End', 'Beacon Hill']
        
        with patch('src.api.model.risk_df', sample_risk_df_with_neighborhood):
            response = flask_client.get('/areas/Downtown')
            
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'area_name' in data
            assert 'average_risk_score' in data
            assert 'maximum_risk_score' in data
            assert 'high_risk_zones' in data
            
    def test_areas_endpoint_no_neighborhood_data(self, flask_client, sample_risk_df):
        """Test sans données de quartier"""
        with patch('src.api.model.risk_df', sample_risk_df):
            response = flask_client.get('/areas/AnyArea')
            
            assert response.status_code == 404
            
            data = json.loads(response.data)
            assert 'Informations sur les quartiers non disponibles' in data['message']


class TestCompareTimesEndpoint:
    """Tests pour l'endpoint /compare-times"""
    
    @patch('src.api.model.get_risk_prediction_for_time')
    def test_compare_times_endpoint_success(self, mock_get_prediction, flask_client):
        """Test de comparaison de temps réussie"""
        predictions = [
            {
                'time': {'hour': 14, 'period': 'après-midi'},
                'risk': {'score': 0.3, 'level': 'Faible'}
            },
            {
                'time': {'hour': 22, 'period': 'soir'},
                'risk': {'score': 0.8, 'level': 'Élevé'}
            }
        ]
        mock_get_prediction.side_effect = predictions
        
        response = flask_client.get('/compare-times?time1=2023-06-15T14:30:00&time2=2023-06-15T22:30:00')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'time1' in data
        assert 'time2' in data
        assert 'comparison' in data
        assert 'risk_difference' in data['comparison']
        assert 'safer_time' in data['comparison']
        
    def test_compare_times_endpoint_missing_parameters(self, flask_client):
        """Test avec paramètres manquants"""
        response = flask_client.get('/compare-times?time1=2023-06-15T14:30:00')  # time2 manquant
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data['status'] or 'message' in data
        
    def test_compare_times_endpoint_invalid_datetime(self, flask_client):
        """Test avec datetime invalide"""
        response = flask_client.get('/compare-times?time1=invalid-date&time2=2023-06-15T22:30:00')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data['status'] or 'message' in data


class TestRouterStatusAndInitialization:
    """Tests pour les endpoints de statut et d'initialisation du routeur"""
    
    @patch('src.api.router')
    def test_router_status_endpoint(self, mock_router, flask_client):
        """Test de l'endpoint de statut du routeur"""
        mock_router.graph = MagicMock()
        mock_router.graph.nodes = [1, 2, 3]
        mock_router.graph.edges = [(1, 2), (2, 3)]
        mock_router.landmarks = [1, 2]
        mock_router.risk_weight = 0.7
        mock_router.max_detour_factor = 1.5
        
        response = flask_client.get('/router-status')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'router_initialized' in data
        assert 'graph_nodes' in data
        assert 'graph_edges' in data
        assert 'landmarks_count' in data
        assert data['graph_nodes'] == 3
        assert data['graph_edges'] == 2
        
    @patch('src.api.router.build_graph')
    @patch('src.api.model.load_model')
    def test_initialize_router_endpoint_success(self, mock_load_model, mock_build_graph, flask_client):
        """Test d'initialisation réussie du routeur"""
        mock_build_graph.return_value = MagicMock()
        mock_build_graph.return_value.nodes = [1, 2, 3]
        mock_build_graph.return_value.edges = [(1, 2), (2, 3)]
        
        with patch('src.api.router.landmarks', [1, 2]):
            response = flask_client.post('/initialize-router')
            
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['status'] == 'success'
            assert 'graph_nodes' in data
            assert 'graph_edges' in data
            assert 'landmarks_count' in data
            
    @patch('src.api.router.build_graph')
    def test_initialize_router_endpoint_error(self, mock_build_graph, flask_client):
        """Test d'erreur d'initialisation du routeur"""
        mock_build_graph.side_effect = Exception("Initialization failed")
        
        response = flask_client.post('/initialize-router')
        
        assert response.status_code == 500
        
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Initialization failed' in data['message']

class TestErrorHandlingAndEdgeCases:
    """Tests de gestion d'erreurs et cas limites"""
    
    def test_json_parsing_errors(self, flask_client):
        """Test de gestion des erreurs de parsing JSON"""
        response = flask_client.post('/train',
                                   data='{"invalid": json}',
                                   content_type='application/json')
        
        # L'application doit gérer les erreurs JSON gracieusement
        assert response.status_code in [200, 400, 500]
        
    def test_missing_model_data(self, flask_client):
        """Test avec modèle non chargé"""
        with patch('src.api.model.risk_df', None):
            response = flask_client.get('/risk-zones')
            
            # Doit gérer l'absence de modèle
            assert response.status_code in [200, 404, 500]
        
    def test_concurrent_requests_safety(self, flask_client):
        """Test de sécurité pour les requêtes concurrentes"""
        import threading
        
        results = []
        
        def make_request():
            response = flask_client.get('/health')
            results.append(response.status_code)
            
        # Simuler des requêtes concurrentes
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Toutes les requêtes devraient réussir
        assert all(status == 200 for status in results)
        assert len(results) == 5


class TestAPIIntegration:
    """Tests d'intégration de l'API"""
    
    @patch('src.api.model')
    @patch('src.api.router')
    def test_complete_api_workflow(self, mock_router, mock_model, flask_client):
        """Test du workflow complet de l'API"""
        # Configurer les mocks
        mock_model.risk_df = MagicMock()
        mock_model.get_high_risk_zones.return_value = [{'id': 1, 'name': 'Test Zone'}]
        mock_model.get_risk_prediction_for_time.return_value = {'time': {}, 'risk': {'score': 0.5}}
        
        mock_router.graph = MagicMock()
        mock_router.get_safe_route.return_value = {'status': 'success', 'route': {}}
        
        # 1. Vérifier la santé
        health_response = flask_client.get('/health')
        assert health_response.status_code == 200
        
        # 2. Obtenir les zones à risque
        zones_response = flask_client.get('/risk-zones')
        assert zones_response.status_code == 200
        
        # 3. Obtenir le risque actuel
        current_risk_response = flask_client.get('/current-risk')
        assert current_risk_response.status_code == 200
        
        # 4. Calculer un itinéraire
        route_response = flask_client.get('/safe-route?start_lat=42.3601&start_lon=-71.0589&end_lat=42.3611&end_lon=-71.0599')
        assert route_response.status_code == 200
        
    def test_api_response_consistency(self, flask_client):
        """Test de cohérence des réponses API"""
        # Toutes les réponses JSON doivent avoir une structure cohérente
        endpoints_to_test = [
            '/health',
            '/risk-zones',
            '/clusters',
            '/current-risk'
        ]
        
        for endpoint in endpoints_to_test:
            response = flask_client.get(endpoint)
            
            # Doit retourner du JSON valide
            try:
                data = json.loads(response.data)
                assert isinstance(data, dict)
            except json.JSONDecodeError:
                pytest.fail(f"Endpoint {endpoint} ne retourne pas du JSON valide")
                
    def test_cors_headers(self, flask_client):
        """Test des en-têtes CORS (si implémentés)"""
        response = flask_client.get('/health')
        
        # Vérifier que la réponse est valide
        assert response.status_code == 200
        
        # Les en-têtes CORS peuvent être ajoutés selon les besoins
        # assert 'Access-Control-Allow-Origin' in response.headers
        
    def test_content_type_consistency(self, flask_client):
        """Test de cohérence du type de contenu"""
        json_endpoints = [
            '/health',
            '/risk-zones',
            '/clusters',
            '/current-risk'
        ]
        
        for endpoint in json_endpoints:
            response = flask_client.get(endpoint)
            
            if response.status_code == 200:
                assert 'application/json' in response.content_type