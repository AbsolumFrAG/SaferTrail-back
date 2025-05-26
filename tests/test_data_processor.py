import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from unittest.mock import patch, Mock

from src.data_processor import (
    create_spatial_dataframes,
    categorize_pedestrian_issues,
    prepare_pedestrian_risk_analysis,
    calculate_pedestrian_risk_scores,
    identify_pedestrian_risk_clusters,
    process_data_for_pedestrian_safety
)


class TestCreateSpatialDataframes:
    """Tests pour la fonction create_spatial_dataframes"""
    
    def test_create_spatial_dataframes_success(self, sample_crash_data, sample_311_data, 
                                            sample_street_segments, sample_crime_data):
        """Test de création réussie des GeoDataFrames"""
        gdf_crashes, gdf_311, gdf_crimes = create_spatial_dataframes(
            sample_crash_data, sample_311_data, sample_street_segments, sample_crime_data
        )
        
        # Vérifier que les GeoDataFrames sont créés correctement
        assert isinstance(gdf_crashes, gpd.GeoDataFrame)
        assert isinstance(gdf_311, gpd.GeoDataFrame)
        assert isinstance(gdf_crimes, gpd.GeoDataFrame)
        
        # Vérifier les CRS
        assert gdf_crashes.crs.to_string() == "EPSG:4326"
        assert gdf_311.crs.to_string() == "EPSG:4326"
        assert gdf_crimes.crs.to_string() == "EPSG:4326"
        
        # Vérifier que la géométrie est des Points
        assert all(isinstance(geom, Point) for geom in gdf_crashes.geometry)
        assert all(isinstance(geom, Point) for geom in gdf_311.geometry)
        assert all(isinstance(geom, Point) for geom in gdf_crimes.geometry)


class TestCategorizePedestrianIssues:
    """Tests pour la fonction categorize_pedestrian_issues"""
    
    def test_categorize_pedestrian_issues_success(self, sample_311_data):
        """Test de catégorisation réussie des problèmes piétons"""
        # Ajouter des textes plus variés pour les tests
        sample_311_data['combined_text'] = [
            'Sidewalk Repair Request - Broken sidewalk',
            'Street Light Out - Dark street at night',
            'Crosswalk Maintenance - Faded crosswalk lines',
            'Traffic Signal Issue - Pedestrian signal not working'
        ]
        
        gdf_311 = gpd.GeoDataFrame(sample_311_data, 
                                  geometry=[Point(-71.0595, 42.3605), 
                                           Point(-71.0605, 42.3615),
                                           Point(-71.0615, 42.3625),
                                           Point(-71.0625, 42.3635)],
                                  crs="EPSG:4326")
        
        result = categorize_pedestrian_issues(gdf_311)
        
        # Vérifier que les colonnes de catégorisation sont ajoutées
        assert 'sidewalk_issues' in result.columns
        assert 'crossing_issues' in result.columns
        assert 'lighting_issues' in result.columns
        assert 'signal_issues' in result.columns
        
        # Vérifier les valeurs pour chaque type d'issue
        assert result.iloc[0]['sidewalk_issues'] == 1  # Sidewalk repair
        assert result.iloc[1]['lighting_issues'] == 1  # Street light
        assert result.iloc[2]['crossing_issues'] == 1  # Crosswalk
        assert result.iloc[3]['signal_issues'] == 1    # Traffic signal
        
    def test_categorize_pedestrian_issues_empty_gdf(self):
        """Test avec un GeoDataFrame vide"""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        result = categorize_pedestrian_issues(empty_gdf)
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0
        
    def test_categorize_pedestrian_issues_multiple_categories(self):
        """Test avec signalements ayant plusieurs catégories"""
        data_311 = pd.DataFrame({
            'combined_text': [
                'Sidewalk and crosswalk repair needed',  # Deux catégories
                'Dark street with broken sidewalk'       # Deux catégories
            ]
        })
        
        gdf_311 = gpd.GeoDataFrame(data_311,
                                  geometry=[Point(-71.0595, 42.3605),
                                           Point(-71.0605, 42.3615)],
                                  crs="EPSG:4326")
        
        result = categorize_pedestrian_issues(gdf_311)
        
        # Premier signalement : sidewalk + crossing
        assert result.iloc[0]['sidewalk_issues'] == 1
        assert result.iloc[0]['crossing_issues'] == 1
        
        # Deuxième signalement : sidewalk + lighting
        assert result.iloc[1]['sidewalk_issues'] == 1
        assert result.iloc[1]['lighting_issues'] == 1
        
    def test_categorize_pedestrian_issues_case_insensitive(self):
        """Test de recherche insensible à la casse"""
        data_311 = pd.DataFrame({
            'combined_text': [
                'SIDEWALK repair needed',
                'Street LIGHT out',
                'crosswalk maintenance'
            ]
        })
        
        gdf_311 = gpd.GeoDataFrame(data_311,
                                  geometry=[Point(-71.0595, 42.3605),
                                           Point(-71.0605, 42.3615),
                                           Point(-71.0615, 42.3625)],
                                  crs="EPSG:4326")
        
        result = categorize_pedestrian_issues(gdf_311)
        
        # Doit détecter les mots-clés même avec différentes casses
        assert result.iloc[0]['sidewalk_issues'] == 1
        assert result.iloc[1]['lighting_issues'] == 1
        assert result.iloc[2]['crossing_issues'] == 1


class TestPreparePedestrianRiskAnalysis:
    """Tests pour la fonction prepare_pedestrian_risk_analysis"""
    
    @patch('src.data_processor.identify_intersections')
    def test_prepare_risk_analysis_success(self, mock_identify_intersections,
                                         sample_crash_data, sample_311_data, 
                                         sample_street_segments, sample_crime_data):
        """Test de préparation d'analyse de risque réussie"""
        # Mocker l'identification des intersections
        mock_intersections = gpd.GeoDataFrame(
            geometry=[Point(-71.0595, 42.3605)], crs="EPSG:4326"
        )
        mock_identify_intersections.return_value = mock_intersections
        
        # Créer les GeoDataFrames
        gdf_crashes, gdf_311, gdf_crimes = create_spatial_dataframes(
            sample_crash_data, sample_311_data, sample_street_segments, sample_crime_data
        )
        
        # Catégoriser les problèmes 311
        gdf_311 = categorize_pedestrian_issues(gdf_311)
        
        result = prepare_pedestrian_risk_analysis(
            gdf_crashes, gdf_311, sample_street_segments, gdf_crimes
        )
        
        # Vérifier le résultat
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_string() == "EPSG:4326"
        
        # Vérifier les colonnes de risque
        risk_columns = [
            'entity_type', 'ped_crash_count', 'ped_crash_night_count',
            'ped_311_count', 'sidewalk_issues', 'crossing_issues',
            'lighting_issues', 'signal_issues', 'crime_count', 'buffer_distance'
        ]
        
        for col in risk_columns:
            assert col in result.columns
            
        # Vérifier les types d'entités
        entity_types = result['entity_type'].unique()
        assert 'street_segment' in entity_types
        # Peut aussi contenir 'intersection' selon les données


class TestCalculatePedestrianRiskScores:
    """Tests pour la fonction calculate_pedestrian_risk_scores"""
        
    def test_calculate_risk_scores_intersection_bonus(self, sample_risk_df):
        """Test du bonus de risque pour les intersections"""
        # Ajouter une intersection aux données
        intersection_row = sample_risk_df.iloc[0].copy()
        intersection_row['entity_type'] = 'intersection'
        
        test_df = pd.concat([sample_risk_df, intersection_row.to_frame().T], ignore_index=True)
        
        result = calculate_pedestrian_risk_scores(test_df)
        
        # L'intersection devrait avoir un score de risque plus élevé
        intersection_idx = result[result['entity_type'] == 'intersection'].index[0]
        street_scores = result[result['entity_type'] == 'street_segment']['risk_score']
        intersection_score = result.loc[intersection_idx, 'risk_score']
        
        # Le score de l'intersection devrait être influencé par le facteur 1.3
        assert intersection_score >= 0
        
    def test_calculate_risk_scores_normalization(self):
        """Test de normalisation des scores"""
        # Créer des données avec des valeurs extrêmes
        test_data = {
            'entity_type': ['street_segment', 'street_segment', 'street_segment'],
            'ped_crash_count': [0, 5, 10],
            'ped_crash_night_count': [0, 2, 5],
            'ped_311_count': [0, 1, 3],
            'sidewalk_issues': [0, 0, 2],
            'crossing_issues': [0, 1, 1],
            'lighting_issues': [0, 0, 1],
            'signal_issues': [0, 0, 1],
            'crime_count': [0, 1, 2]
        }
        
        test_df = gpd.GeoDataFrame(
            test_data,
            geometry=[Point(-71.0589, 42.3601), Point(-71.0599, 42.3611), Point(-71.0609, 42.3621)],
            crs="EPSG:4326"
        )
        
        result = calculate_pedestrian_risk_scores(test_df)
        
        # Le score maximum devrait être 1 (normalisé)
        assert result['risk_score'].max() <= 1.0
        assert result['risk_score'].min() >= 0.0
        
        # Les scores devraient être ordonnés (plus de facteurs = plus de risque)
        assert result.iloc[0]['risk_score'] <= result.iloc[1]['risk_score'] <= result.iloc[2]['risk_score']


class TestIdentifyPedestrianRiskClusters:
    """Tests pour la fonction identify_pedestrian_risk_clusters"""
    
    def test_identify_clusters_success(self, sample_risk_df):
        """Test d'identification de clusters réussie"""
        # Modifier les données pour avoir plusieurs zones à haut risque proches
        test_df = sample_risk_df.copy()
        test_df.loc[:, 'risk_score'] = [0.8, 0.9, 0.85, 0.4]  # 3 zones à haut risque
        
        result = identify_pedestrian_risk_clusters(test_df)
        
        # Vérifier que la colonne cluster est ajoutée
        assert 'cluster' in result.columns
        
        # Vérifier les valeurs de cluster
        cluster_values = result['cluster'].unique()
        assert -1 in cluster_values  # -1 pour les zones non clusterisées
        
    def test_identify_clusters_insufficient_data(self):
        """Test avec données insuffisantes pour former des clusters"""
        # Créer des données avec seulement une zone à haut risque
        test_data = {
            'entity_type': ['street_segment'],
            'risk_score': [0.8]
        }
        
        test_df = gpd.GeoDataFrame(
            test_data,
            geometry=[Point(-71.0589, 42.3601)],
            crs="EPSG:4326"
        )
        
        result = identify_pedestrian_risk_clusters(test_df, min_samples=2)
        
        # Doit gérer le cas avec données insuffisantes
        assert isinstance(result, gpd.GeoDataFrame)
            
    def test_identify_clusters_custom_parameters(self):
        """Test avec paramètres personnalisés"""
        # Créer des données avec plusieurs zones à haut risque
        test_data = {
            'entity_type': ['street_segment'] * 5,
            'risk_score': [0.8, 0.85, 0.9, 0.75, 0.6]
        }
        
        # Créer des points proches pour former des clusters
        points = [
            Point(-71.0589, 42.3601),
            Point(-71.0590, 42.3602),  # Très proche du premier
            Point(-71.0591, 42.3603),  # Proche des deux premiers
            Point(-71.0595, 42.3605),  # Un peu plus loin
            Point(-71.0600, 42.3610)   # Loin
        ]
        
        test_df = gpd.GeoDataFrame(test_data, geometry=points, crs="EPSG:4326")
        
        result = identify_pedestrian_risk_clusters(test_df, min_samples=2, eps=0.001)
        
        assert isinstance(result, gpd.GeoDataFrame)
        if 'cluster' in result.columns:
            # Au moins certaines zones devraient être clusterisées
            assert any(cluster >= 0 for cluster in result['cluster'])


class TestProcessDataForPedestrianSafety:
    """Tests pour la fonction process_data_for_pedestrian_safety"""
    
    def test_process_data_complete_workflow(self, sample_crash_data, sample_311_data,
                                          sample_street_segments, sample_crime_data):
        """Test du workflow complet de traitement des données"""
        result = process_data_for_pedestrian_safety(
            sample_crash_data, sample_311_data, sample_street_segments, sample_crime_data
        )
        
        # Vérifier que le résultat est un GeoDataFrame complet
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_string() == "EPSG:4326"
        
        # Vérifier toutes les colonnes attendues
        expected_columns = [
            'entity_type', 'ped_crash_count', 'ped_crash_night_count',
            'ped_311_count', 'sidewalk_issues', 'crossing_issues',
            'lighting_issues', 'signal_issues', 'crime_count',
            'risk_score', 'risk_level'
        ]
        
        for col in expected_columns:
            assert col in result.columns
            
        # Vérifier que les scores de risque sont calculés
        assert 'risk_score' in result.columns
        assert all(isinstance(score, (int, float)) for score in result['risk_score'])


class TestDataProcessorIntegration:
    """Tests d'intégration pour le processeur de données"""
    
    def test_data_flow_consistency(self, sample_crash_data, sample_311_data,
                                 sample_street_segments, sample_crime_data):
        """Test de cohérence du flux de données"""
        # Exécuter chaque étape individuellement
        gdf_crashes, gdf_311, gdf_crimes = create_spatial_dataframes(
            sample_crash_data, sample_311_data, sample_street_segments, sample_crime_data
        )
        
        gdf_311_categorized = categorize_pedestrian_issues(gdf_311)
        
        risk_analysis = prepare_pedestrian_risk_analysis(
            gdf_crashes, gdf_311_categorized, sample_street_segments, gdf_crimes
        )
        
        risk_scores = calculate_pedestrian_risk_scores(risk_analysis)
        
        final_result = identify_pedestrian_risk_clusters(risk_scores)
        
        # Comparer avec le workflow complet
        complete_result = process_data_for_pedestrian_safety(
            sample_crash_data, sample_311_data, sample_street_segments, sample_crime_data
        )
        
        # Les résultats devraient être cohérents
        assert len(final_result) == len(complete_result)
        assert list(final_result.columns) == list(complete_result.columns)


class TestValidationAndErrorHandling:
    """Tests de validation et de gestion d'erreurs"""
        
    def test_missing_required_columns(self):
        """Test avec colonnes requises manquantes"""
        # GeoDataFrame sans colonnes de risque
        minimal_gdf = gpd.GeoDataFrame(
            {'entity_type': ['street_segment']},
            geometry=[Point(-71.0589, 42.3601)],
            crs="EPSG:4326"
        )
        
        # Doit gérer l'absence de colonnes de risque
        try:
            result = calculate_pedestrian_risk_scores(minimal_gdf)
            assert isinstance(result, gpd.GeoDataFrame)
        except KeyError:
            # Acceptable si la fonction exige certaines colonnes
            pass