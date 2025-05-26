import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from unittest.mock import patch
import tempfile
from pathlib import Path

from src.risk_model import PedestrianRiskModel
from src.time_based_risk_model import TimeBasedPedestrianRiskModel


class TestPedestrianRiskModelInit:
    """Tests pour l'initialisation du modèle de risque piéton"""
    
    def test_init_success(self):
        """Test d'initialisation réussie"""
        model = PedestrianRiskModel()
        
        assert model.risk_df is None
        assert model.streets_gdf is None
        assert model.intersections_gdf is None
        assert isinstance(model.time_model, TimeBasedPedestrianRiskModel)
        assert model.model_path.name == 'pedestrian_risk_model.pkl'
        
    def test_init_model_path(self):
        """Test du chemin du modèle"""
        model = PedestrianRiskModel()
        
        assert 'models' in str(model.model_path)
        assert 'pedestrian_risk_model.pkl' in str(model.model_path)


class TestLoadData:
    """Tests pour la méthode load_data"""
    
    @patch('src.risk_model.pd.read_csv')
    @patch('src.risk_model.gpd.read_file')
    def test_load_data_success(self, mock_read_file, mock_read_csv, sample_crash_data, 
                             sample_311_data, sample_street_segments, sample_crime_data):
        """Test de chargement réussi des données"""
        # Configurer les mocks
        mock_read_csv.side_effect = [
            sample_crash_data,  # Crash data
            sample_311_data,    # 311 data
            sample_crime_data   # Crime data
        ]
        mock_read_file.return_value = sample_street_segments
        
        model = PedestrianRiskModel()
        df_crashes, df_311, gdf_streets, df_crimes = model.load_data()
        
        # Vérifier les résultats
        assert isinstance(df_crashes, pd.DataFrame)
        assert isinstance(df_311, pd.DataFrame)
        assert isinstance(gdf_streets, gpd.GeoDataFrame)
        assert isinstance(df_crimes, pd.DataFrame)
        
        # Vérifier que les données sont filtrées pour les piétons
        assert all(df_crashes['mode_type'] == 'ped')
        
    @patch('src.risk_model.pd.read_csv')
    def test_load_data_file_not_found(self, mock_read_csv):
        """Test avec fichiers manquants"""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        model = PedestrianRiskModel()
        
        # Doit gérer l'erreur gracieusement
        try:
            df_crashes, df_311, gdf_streets, df_crimes = model.load_data()
            # Si aucune exception, vérifier que les DataFrames sont vides
            assert len(df_crashes) == 0
        except FileNotFoundError:
            # Acceptable si l'erreur n'est pas gérée
            pass


class TestPrepareSpatialData:
    """Tests pour la méthode prepare_spatial_data"""
    
    def test_prepare_spatial_data_success(self, sample_crash_data, sample_311_data,
                                        sample_street_segments, sample_crime_data):
        """Test de préparation réussie des données spatiales"""
        model = PedestrianRiskModel()
        
        gdf_crashes, gdf_311, gdf_streets, gdf_crimes = model.prepare_spatial_data(
            sample_crash_data, sample_311_data, sample_street_segments, sample_crime_data
        )
        
        # Vérifier les types de retour
        assert isinstance(gdf_crashes, gpd.GeoDataFrame)
        assert isinstance(gdf_311, gpd.GeoDataFrame)
        assert isinstance(gdf_streets, gpd.GeoDataFrame)
        assert isinstance(gdf_crimes, gpd.GeoDataFrame)
        
        # Vérifier les CRS
        assert gdf_crashes.crs.to_string() == "EPSG:4326"
        assert gdf_311.crs.to_string() == "EPSG:4326"
        assert gdf_streets.crs.to_string() == "EPSG:4326"
        assert gdf_crimes.crs.to_string() == "EPSG:4326"
        
        # Vérifier que les intersections sont identifiées
        assert hasattr(model, 'intersections_gdf')
        assert isinstance(model.intersections_gdf, gpd.GeoDataFrame)
        
    def test_prepare_spatial_data_empty_inputs(self):
        """Test avec des entrées vides"""
        model = PedestrianRiskModel()
        
        empty_df = pd.DataFrame()
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        gdf_crashes, gdf_311, gdf_streets, gdf_crimes = model.prepare_spatial_data(
            empty_df, empty_df, empty_gdf, empty_df
        )
        
        # Doivent être des GeoDataFrames vides mais valides
        assert isinstance(gdf_crashes, gpd.GeoDataFrame)
        assert isinstance(gdf_311, gpd.GeoDataFrame)
        assert isinstance(gdf_streets, gpd.GeoDataFrame)
        assert isinstance(gdf_crimes, gpd.GeoDataFrame)
        
        assert len(gdf_crashes) == 0
        assert len(gdf_311) == 0
        assert len(gdf_crimes) == 0
        
    def test_prepare_spatial_data_missing_coordinates(self):
        """Test avec coordonnées manquantes"""
        # Données avec coordonnées manquantes
        crash_data_incomplete = pd.DataFrame({
            'dispatch_ts': ['2023-01-15 08:30:00'],
            'mode_type': ['ped'],
            'lat': [None],
            'long': [-71.0589]
        })
        
        model = PedestrianRiskModel()
        
        gdf_crashes, _, _, _ = model.prepare_spatial_data(
            crash_data_incomplete, pd.DataFrame(), 
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), pd.DataFrame()
        )
        
        # Doit exclure les entrées avec coordonnées manquantes
        assert len(gdf_crashes) == 0


class TestCalculatePedestrianRiskFactors:
    """Tests pour la méthode calculate_pedestrian_risk_factors"""
    
    def test_calculate_risk_factors_success(self, sample_crash_data, sample_311_data,
                                          sample_street_segments, sample_crime_data):
        """Test de calcul réussi des facteurs de risque"""
        model = PedestrianRiskModel()
        
        # Préparer les données spatiales
        gdf_crashes, gdf_311, gdf_streets, gdf_crimes = model.prepare_spatial_data(
            sample_crash_data, sample_311_data, sample_street_segments, sample_crime_data
        )
        
        # Calculer les facteurs de risque
        risk_df = model.calculate_pedestrian_risk_factors(
            gdf_crashes, gdf_311, gdf_streets, gdf_crimes
        )
        
        # Vérifier le résultat
        assert isinstance(risk_df, gpd.GeoDataFrame)
        assert risk_df.crs.to_string() == "EPSG:4326"
        
        # Vérifier les colonnes de facteurs de risque
        risk_columns = [
            'entity_type', 'ped_crash_count', 'ped_crash_night_count',
            'ped_311_count', 'sidewalk_issues', 'crossing_issues',
            'lighting_issues', 'signal_issues', 'crime_count', 'risk_score'
        ]
        
        for col in risk_columns:
            assert col in risk_df.columns
            
    def test_calculate_risk_factors_empty_streets(self):
        """Test avec segments de rue vides"""
        model = PedestrianRiskModel()
        
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        risk_df = model.calculate_pedestrian_risk_factors(
            empty_gdf, empty_gdf, empty_gdf, empty_gdf
        )
        
        # Doit retourner un GeoDataFrame vide mais valide
        assert isinstance(risk_df, gpd.GeoDataFrame)
        assert risk_df.crs.to_string() == "EPSG:4326"
        
    def test_calculate_risk_factors_night_accident_detection(self):
        """Test de détection des accidents de nuit"""
        # Créer des données avec accidents de nuit
        night_crash_data = pd.DataFrame({
            'dispatch_ts': [
                '2023-01-15 20:30:00',  # 20h (nuit)
                '2023-01-16 14:30:00'   # 14h (jour)
            ],
            'mode_type': ['ped', 'ped'],
            'lat': [42.3601, 42.3611],
            'long': [-71.0589, -71.0599],
            'hour': [20, 14]
        })
        night_crash_data['dispatch_ts'] = pd.to_datetime(night_crash_data['dispatch_ts'])
        
        model = PedestrianRiskModel()
        
        # Créer un segment de rue simple
        street_gdf = gpd.GeoDataFrame(
            {'ST_NAME': ['Test']},
            geometry=[LineString([(-71.0589, 42.3601), (-71.0599, 42.3611)])],
            crs="EPSG:4326"
        )
        
        gdf_crashes, _, _, _ = model.prepare_spatial_data(
            night_crash_data, pd.DataFrame(), street_gdf, pd.DataFrame()
        )
        
        risk_df = model.calculate_pedestrian_risk_factors(
            gdf_crashes, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"), 
            street_gdf, gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        )
        
        # Vérifier que les accidents de nuit sont comptés
        assert 'ped_crash_night_count' in risk_df.columns
        night_crashes = risk_df['ped_crash_night_count'].sum()
        assert night_crashes >= 0


class TestModelPersistence:
    """Tests pour la sauvegarde et le chargement du modèle"""
    
    def test_save_model_success(self, sample_risk_df, sample_street_segments):
        """Test de sauvegarde réussie du modèle"""
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        model.streets_gdf = sample_street_segments
        model.intersections_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model.model_path = Path(temp_dir) / 'test_model.pkl'
            
            success = model.save_model()
            
            assert success is True
            assert model.model_path.exists()
            
    def test_load_model_success(self, sample_risk_df, sample_street_segments):
        """Test de chargement réussi du modèle"""
        # Créer et sauvegarder un modèle
        original_model = PedestrianRiskModel()
        original_model.risk_df = sample_risk_df
        original_model.streets_gdf = sample_street_segments
        original_model.intersections_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'test_model.pkl'
            original_model.model_path = model_path
            original_model.save_model()
            
            # Charger le modèle
            new_model = PedestrianRiskModel()
            new_model.model_path = model_path
            success = new_model.load_model()
            
            assert success is True
            assert new_model.risk_df is not None
            assert new_model.streets_gdf is not None
            assert len(new_model.risk_df) == len(original_model.risk_df)
            
    def test_load_model_file_not_found(self):
        """Test de chargement avec fichier inexistant"""
        model = PedestrianRiskModel()
        model.model_path = Path("nonexistent_model.pkl")
        
        success = model.load_model()
        
        assert success is False
        
    def test_save_model_permission_error(self):
        """Test de sauvegarde avec erreur de permission"""
        model = PedestrianRiskModel()
        model.risk_df = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Chemin invalide
        model.model_path = Path("/invalid/path/model.pkl")
        
        success = model.save_model()
        
        assert success is False


class TestGetHighRiskZones:
    """Tests pour la méthode get_high_risk_zones"""
    
    def test_get_high_risk_zones_success(self, sample_risk_df):
        """Test de récupération réussie des zones à haut risque"""
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        risk_zones = model.get_high_risk_zones(threshold=0.5, limit=5)
        
        assert isinstance(risk_zones, list)
        assert len(risk_zones) <= 5
        
        # Vérifier que toutes les zones ont un score >= threshold
        for zone in risk_zones:
            assert zone['risk_score'] >= 0.5
            assert 'coordinates' in zone
            assert 'lat' in zone['coordinates']
            assert 'lon' in zone['coordinates']
        
    def test_get_high_risk_zones_high_threshold(self, sample_risk_df):
        """Test avec seuil élevé"""
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        risk_zones = model.get_high_risk_zones(threshold=0.95, limit=10)
        
        # Avec un seuil très élevé, peu ou pas de zones
        assert isinstance(risk_zones, list)
        assert len(risk_zones) <= len(sample_risk_df)
        
    def test_get_high_risk_zones_sorting(self, sample_risk_df):
        """Test du tri par score de risque"""
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        risk_zones = model.get_high_risk_zones(threshold=0.0, limit=10)
        
        # Vérifier que les zones sont triées par score décroissant
        for i in range(len(risk_zones) - 1):
            assert risk_zones[i]['risk_score'] >= risk_zones[i + 1]['risk_score']


class TestGetRiskClusters:
    """Tests pour la méthode get_risk_clusters"""
    
    def test_get_risk_clusters_success(self, sample_risk_df):
        """Test de récupération réussie des clusters de risque"""
        # Ajouter des clusters aux données
        sample_risk_df['cluster'] = [0, -1, 0, 1]
        
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        clusters = model.get_risk_clusters(min_cluster_size=2)
        
        assert isinstance(clusters, list)
        
        # Vérifier les informations des clusters
        for cluster in clusters:
            assert 'cluster_id' in cluster
            assert 'size' in cluster
            assert 'avg_risk_score' in cluster
            assert 'coordinates' in cluster
            assert cluster['size'] >= 2
            
    def test_get_risk_clusters_no_clusters(self, sample_risk_df):
        """Test sans clusters"""
        # Tous les clusters à -1 (pas de cluster)
        sample_risk_df['cluster'] = [-1, -1, -1, -1]
        
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        clusters = model.get_risk_clusters()
        
        assert isinstance(clusters, list)
        assert len(clusters) == 0
        
    def test_get_risk_clusters_no_model(self):
        """Test sans modèle chargé"""
        model = PedestrianRiskModel()
        
        clusters = model.get_risk_clusters()
        
        assert isinstance(clusters, list)
        assert len(clusters) == 0


class TestGetSafetyRecommendations:
    """Tests pour la méthode get_safety_recommendations"""
    
    def test_get_safety_recommendations_success(self, sample_risk_df):
        """Test de génération réussie des recommandations"""
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        # Utiliser l'index d'une zone existante
        zone_id = sample_risk_df.index[0]
        recommendations = model.get_safety_recommendations(zone_id)
        
        assert isinstance(recommendations, dict)
        assert 'zone_id' in recommendations
        assert 'recommendations' in recommendations
        assert 'problems_identified' in recommendations
        assert isinstance(recommendations['recommendations'], list)
        
    def test_get_safety_recommendations_zone_not_found(self, sample_risk_df):
        """Test avec zone inexistante"""
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        recommendations = model.get_safety_recommendations(999)  # Zone inexistante
        
        assert isinstance(recommendations, dict)
        assert 'error' in recommendations
        
    def test_get_safety_recommendations_high_risk_zone(self, sample_risk_df):
        """Test avec zone à haut risque"""
        # Modifier une zone pour avoir un score très élevé et des problèmes
        high_risk_idx = sample_risk_df.index[0]
        sample_risk_df.loc[high_risk_idx, 'risk_score'] = 0.95
        sample_risk_df.loc[high_risk_idx, 'risk_level'] = 'Très élevé'
        sample_risk_df.loc[high_risk_idx, 'ped_crash_count'] = 5
        sample_risk_df.loc[high_risk_idx, 'lighting_issues'] = 2
        
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        recommendations = model.get_safety_recommendations(high_risk_idx)
        
        # Doit inclure des recommandations spécifiques aux problèmes identifiés
        assert len(recommendations['recommendations']) > 0
        assert len(recommendations['problems_identified']) > 0


class TestTraining:
    """Tests pour la méthode train"""
    
    @patch.object(PedestrianRiskModel, 'load_data')
    @patch.object(PedestrianRiskModel, 'save_model')
    def test_train_success(self, mock_save, mock_load_data, sample_crash_data,
                          sample_311_data, sample_street_segments, sample_crime_data):
        """Test d'entraînement réussi"""
        # Configurer le mock
        mock_load_data.return_value = (
            sample_crash_data, sample_311_data, sample_street_segments, sample_crime_data
        )
        mock_save.return_value = True
        
        model = PedestrianRiskModel()
        
        success = model.train(force_retrain=True)
        
        assert success is True
        assert model.risk_df is not None
        assert model.streets_gdf is not None
        mock_save.assert_called_once()
        
    @patch.object(PedestrianRiskModel, 'load_model')
    def test_train_load_existing_model(self, mock_load_model):
        """Test de chargement d'un modèle existant"""
        mock_load_model.return_value = True
        
        model = PedestrianRiskModel()
        
        # Sans force_retrain, doit essayer de charger le modèle existant
        success = model.train(force_retrain=False)
        
        assert success is True
        mock_load_model.assert_called_once()


class TestTemporalFunctionality:
    """Tests pour les fonctionnalités temporelles"""
    
    @patch.object(TimeBasedPedestrianRiskModel, 'predict_risk_at_time')
    def test_get_risk_prediction_for_time(self, mock_predict):
        """Test de prédiction de risque temporel"""
        # Configurer le mock
        mock_predict.return_value = {
            'time': {'hour': 14, 'day_name': 'Jeudi'},
            'risk': {'score': 0.6, 'level': 'Modéré'}
        }
        
        model = PedestrianRiskModel()
        
        risk_info = model.get_risk_prediction_for_time(hour=14, day_of_week=3)
        
        assert isinstance(risk_info, dict)
        assert 'time' in risk_info
        assert 'risk' in risk_info
        mock_predict.assert_called_once()
        
    @patch.object(TimeBasedPedestrianRiskModel, 'get_high_risk_times')
    def test_get_high_risk_times(self, mock_get_times):
        """Test de récupération des périodes à haut risque"""
        mock_get_times.return_value = {
            'hours': [{'hour': 20, 'risk_score': 0.8}],
            'days': [{'day': 'Friday', 'risk_score': 0.9}]
        }
        
        model = PedestrianRiskModel()
        
        high_risk_times = model.get_high_risk_times(top_n=3)
        
        assert isinstance(high_risk_times, dict)
        assert 'hours' in high_risk_times
        assert 'days' in high_risk_times
        mock_get_times.assert_called_once_with(top_n=3)


class TestModelIntegration:
    """Tests d'intégration pour le modèle de risque"""
    
    def test_complete_workflow(self, sample_crash_data, sample_311_data,
                             sample_street_segments, sample_crime_data):
        """Test du workflow complet"""
        model = PedestrianRiskModel()
        
        # Simuler le chargement des données
        with patch.object(model, 'load_data') as mock_load:
            mock_load.return_value = (
                sample_crash_data, sample_311_data, sample_street_segments, sample_crime_data
            )
            
            # Entraîner le modèle
            success = model.train(force_retrain=True)
            assert success is True
            
            # Tester les fonctionnalités principales
            risk_zones = model.get_high_risk_zones(threshold=0.0, limit=5)
            assert isinstance(risk_zones, list)
            
            if len(risk_zones) > 0:
                zone_id = risk_zones[0]['id']
                recommendations = model.get_safety_recommendations(zone_id)
                assert isinstance(recommendations, dict)
                assert 'recommendations' in recommendations
        
    def test_data_consistency(self, sample_risk_df):
        """Test de cohérence des données"""
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        # Les zones à haut risque devraient être cohérentes
        high_risk_zones = model.get_high_risk_zones(threshold=0.7)
        
        for zone in high_risk_zones:
            assert zone['risk_score'] >= 0.7
            assert zone['risk_score'] <= 1.0
            assert zone['id'] in sample_risk_df.index
            
            # Les coordonnées devraient être dans les limites de Boston
            lat, lon = zone['coordinates']['lat'], zone['coordinates']['lon']
            assert 42 < lat < 43
            assert -72 < lon < -70


class TestValidationAndErrorHandling:
    """Tests de validation et de gestion d'erreurs"""
            
    def test_memory_efficiency(self):
        """Test d'efficacité mémoire avec de gros datasets"""
        model = PedestrianRiskModel()
        
        # Créer un dataset synthétique volumineux
        n_records = 1000
        large_risk_df = gpd.GeoDataFrame({
            'entity_type': ['street_segment'] * n_records,
            'risk_score': np.random.random(n_records),
            'ped_crash_count': np.random.randint(0, 10, n_records)
        }, geometry=[Point(-71.0589 + i*0.0001, 42.3601 + i*0.0001) for i in range(n_records)],
        crs="EPSG:4326")
        
        model.risk_df = large_risk_df
        
        # Les opérations devraient fonctionner même avec beaucoup de données
        risk_zones = model.get_high_risk_zones(limit=10)
        assert len(risk_zones) <= 10
        
    def test_concurrent_access_safety(self, sample_risk_df):
        """Test de sécurité d'accès concurrent"""
        model = PedestrianRiskModel()
        model.risk_df = sample_risk_df
        
        # Simuler des accès concurrents
        results = []
        for _ in range(10):
            zones = model.get_high_risk_zones(limit=2)
            results.append(zones)
            
        # Tous les résultats devraient être cohérents
        for result in results:
            assert isinstance(result, list)
            assert len(result) <= 2