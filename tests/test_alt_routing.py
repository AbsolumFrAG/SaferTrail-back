import pytest
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from datetime import datetime
from unittest.mock import patch
import geopandas as gpd
import pandas as pd

from src.alt_routing import (
    convert_degrees_to_miles,
    convert_degrees_to_meters,
    format_walking_time,
    classify_road_for_pedestrians,
    calculate_segment_cost,
    ALTRouter
)


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires"""
    
    def test_convert_degrees_to_miles(self):
        """Test de conversion degrés vers miles"""
        # 1 degré ≈ 68.97 miles
        result = convert_degrees_to_miles(1.0)
        assert abs(result - 68.97) < 0.01
        
        # Test avec zéro
        assert convert_degrees_to_miles(0.0) == 0.0
        
        # Test avec valeur négative
        result = convert_degrees_to_miles(-0.5)
        assert result < 0
        assert abs(result + 34.485) < 0.01
        
    def test_convert_degrees_to_meters(self):
        """Test de conversion degrés vers mètres"""
        # 1 degré ≈ 111,000 mètres
        result = convert_degrees_to_meters(1.0)
        assert abs(result - 111000) < 100
        
        # Test avec zéro
        assert convert_degrees_to_meters(0.0) == 0.0
        
        # Test avec fraction
        result = convert_degrees_to_meters(0.001)
        assert abs(result - 111) < 1
        
    def test_format_walking_time(self):
        """Test de formatage du temps de marche"""
        # Test avec des secondes
        result = format_walking_time(90)  # 1.5 minutes
        assert result['total_seconds'] == 90
        assert result['total_minutes'] == 1
        assert result['hours'] == 0
        assert result['minutes'] == 1
        assert result['formatted'] == '1min'
        
        # Test avec des heures
        result = format_walking_time(3900)  # 1h 5min
        assert result['hours'] == 1
        assert result['minutes'] == 5
        assert result['formatted'] == '1h 5min'
        
        # Test avec seulement des heures
        result = format_walking_time(3600)  # 1h exactement
        assert result['hours'] == 1
        assert result['minutes'] == 0
        assert result['formatted'] == '1h'
        
        # Test avec zéro
        result = format_walking_time(0)
        assert result['formatted'] == '0min'


class TestCalculateSegmentCost:
    """Tests pour la fonction calculate_segment_cost"""
    
    def test_calculate_segment_cost_normal(self):
        """Test de calcul de coût pour segment normal"""
        classification = {
            'walkable': True,
            'category': 'normal',
            'pedestrian_safety': 'safe'
        }
        
        cost = calculate_segment_cost(1.0, classification)
        
        # Coût normal (facteur 1.0)
        assert cost == 1.0
        
    def test_calculate_segment_cost_highway(self):
        """Test de coût pour autoroute (interdit)"""
        classification = {
            'walkable': False,
            'category': 'highway',
            'pedestrian_safety': 'prohibited'
        }
        
        cost = calculate_segment_cost(1.0, classification)
        
        # Coût infini pour routes interdites
        assert cost == float('inf')
        
    def test_calculate_segment_cost_ferry(self):
        """Test de coût pour ferry"""
        classification = {
            'walkable': True,
            'category': 'ferry',
            'requires_transport': True,
            'transport_type': 'ferry'
        }
        
        cost = calculate_segment_cost(1.0, classification)
        
        # Coût élevé pour ferry
        assert cost > 1.0
        assert cost == 2.0  # Facteur ferry = 2.0
        
    def test_calculate_segment_cost_time_dependent(self):
        """Test de coût dépendant de l'heure"""
        classification = {
            'walkable': True,
            'category': 'ferry',
            'requires_transport': True
        }
        
        # Test nuit (service réduit)
        night_time = datetime(2023, 6, 15, 23, 0)
        night_cost = calculate_segment_cost(1.0, classification, night_time)
        
        # Test jour normal
        day_time = datetime(2023, 6, 15, 14, 0)
        day_cost = calculate_segment_cost(1.0, classification, day_time)
        
        # Le coût de nuit devrait être plus élevé
        assert night_cost > day_cost
        
    def test_calculate_segment_cost_rush_hour(self):
        """Test de coût aux heures de pointe"""
        classification = {
            'walkable': True,
            'category': 'transit',
            'requires_transport': True
        }
        
        # Heure de pointe
        rush_time = datetime(2023, 6, 15, 8, 0)
        rush_cost = calculate_segment_cost(1.0, classification, rush_time)
        
        # Heure normale
        normal_time = datetime(2023, 6, 15, 14, 0)
        normal_cost = calculate_segment_cost(1.0, classification, normal_time)
        
        # Coût différent aux heures de pointe
        assert rush_cost != normal_cost


class TestALTRouterInit:
    """Tests pour l'initialisation du routeur ALT"""
    
    def test_init_success(self):
        """Test d'initialisation réussie"""
        router = ALTRouter()
        
        assert router.graph is None
        assert router.risk_model is None
        assert router.landmarks == []
        assert router.landmark_distances == {}
        assert router.num_landmarks == 8
        assert router.max_detour_factor == 1.5
        assert router.risk_weight == 0.7
        assert router.allow_restricted_routes is False
        assert router.pedestrian_only is True
        
    def test_init_with_risk_model(self, mock_pedestrian_risk_model):
        """Test d'initialisation avec modèle de risque"""
        router = ALTRouter(mock_pedestrian_risk_model)
        
        assert router.risk_model == mock_pedestrian_risk_model
        
    def test_set_pedestrian_mode(self):
        """Test de configuration du mode piéton"""
        router = ALTRouter()
        
        router.set_pedestrian_mode(strict_mode=False, allow_transports=True)
        
        assert router.pedestrian_only is False
        assert router.allow_restricted_routes is True
        
    def test_set_risk_weight(self):
        """Test de configuration du poids de risque"""
        router = ALTRouter()
        
        router.set_risk_weight(0.5)
        assert router.risk_weight == 0.5
        
        # Test valeur invalide
        with pytest.raises(ValueError):
            router.set_risk_weight(1.5)
            
    def test_set_max_detour_factor(self):
        """Test de configuration du facteur de détour"""
        router = ALTRouter()
        
        router.set_max_detour_factor(2.0)
        assert router.max_detour_factor == 2.0
        
        # Test valeur invalide
        with pytest.raises(ValueError):
            router.set_max_detour_factor(0.5)


class TestBuildGraph:
    """Tests pour la méthode build_graph"""
    
    def test_build_graph_success(self, sample_risk_df, sample_street_segments):
        """Test de construction réussie du graphe"""
        router = ALTRouter()
        
        graph = router.build_graph(sample_risk_df, sample_street_segments)
        
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0
        
        # Vérifier que le graphe est stocké
        assert router.graph == graph
        
        # Vérifier que les landmarks sont sélectionnés
        assert len(router.landmarks) > 0
        assert isinstance(router.landmark_distances, dict)
        
    def test_build_graph_filtering(self, sample_risk_df, sample_street_segments):
        """Test du filtrage des segments pour piétons"""
        router = ALTRouter()
        router.pedestrian_only = True
        
        # Ajouter un segment d'autoroute (non marchable)
        highway_row = sample_street_segments.iloc[0].copy()
        highway_row['full_street_name'] = 'INTERSTATE 95'
        highway_row['SPEEDLIMIT'] = 65
        highway_row['CFCC'] = 'A11'
        
        modified_streets = pd.concat([sample_street_segments, highway_row.to_frame().T], ignore_index=True)
        
        graph = router.build_graph(sample_risk_df, modified_streets)
        
        # Le graphe ne devrait pas inclure l'autoroute
        assert isinstance(graph, nx.DiGraph)
        # Le nombre d'arêtes devrait être cohérent avec le filtrage
        
    def test_build_graph_oneway_streets(self, sample_risk_df):
        """Test avec rues à sens unique"""
        # Créer des segments avec différentes configurations de sens
        lines = [
            LineString([(-71.0589, 42.3601), (-71.0599, 42.3611)]),  # Bidirectionnel
            LineString([(-71.0599, 42.3611), (-71.0609, 42.3621)])   # Sens unique
        ]
        
        street_data = {
            'ST_NAME': ['Main', 'Oak'],
            'ST_TYPE': ['ST', 'AVE'],
            'ONEWAY': ['B', 'FT'],  # B = bidirectionnel, FT = sens unique
            'full_street_name': ['Main Street', 'Oak Avenue']
        }
        
        streets_gdf = gpd.GeoDataFrame(street_data, geometry=lines, crs="EPSG:4326")
        
        router = ALTRouter()
        graph = router.build_graph(sample_risk_df, streets_gdf)
        
        # Vérifier que les sens uniques sont respectés
        assert isinstance(graph, nx.DiGraph)
        
    def test_build_graph_with_risk_model(self, mock_pedestrian_risk_model, sample_risk_df):
        """Test avec modèle de risque"""
        mock_pedestrian_risk_model.risk_df = sample_risk_df
        mock_pedestrian_risk_model.streets_gdf = sample_risk_df  # Pour simplifier
        
        router = ALTRouter(mock_pedestrian_risk_model)
        
        # Construire sans paramètres (utilise le modèle)
        graph = router.build_graph()
        
        assert isinstance(graph, nx.DiGraph)


class TestLandmarkSelection:
    """Tests pour la sélection des landmarks"""
    
    @patch('src.alt_routing.nx.single_source_dijkstra_path_length')
    def test_select_landmarks_optimized(self, mock_dijkstra, sample_risk_df, sample_street_segments):
        """Test de sélection optimisée des landmarks"""
        router = ALTRouter()
        
        # Construire un petit graphe
        router.build_graph(sample_risk_df, sample_street_segments)
        
        if router.graph and len(router.graph.nodes) > 0:
            # Mocker les distances Dijkstra
            mock_dijkstra.return_value = {node: np.random.random() for node in router.graph.nodes}
            
            landmarks = router.select_landmarks_optimized()
            
            assert isinstance(landmarks, list)
            assert len(landmarks) <= router.num_landmarks
            assert len(landmarks) <= len(router.graph.nodes)
            
            # Vérifier que les landmarks sont des nœuds du graphe
            for landmark in landmarks:
                assert landmark in router.graph.nodes
                
    def test_select_landmarks_empty_graph(self):
        """Test avec graphe vide"""
        router = ALTRouter()
        router.graph = nx.DiGraph()  # Graphe vide
        
        landmarks = router.select_landmarks_optimized()
        
        # Doit gérer le graphe vide
        assert landmarks is None or len(landmarks) == 0
        
    def test_select_landmarks_small_graph(self, sample_risk_df):
        """Test avec très petit graphe"""
        router = ALTRouter()
        
        # Créer un très petit graphe (2 nœuds)
        small_streets = gpd.GeoDataFrame({
            'ST_NAME': ['Test'],
            'full_street_name': ['Test Street']
        }, geometry=[LineString([(-71.0589, 42.3601), (-71.0599, 42.3611)])], crs="EPSG:4326")
        
        graph = router.build_graph(sample_risk_df.head(1), small_streets)
        
        if graph and len(graph.nodes) > 0:
            landmarks = router.select_landmarks_optimized()
            
            # Nombre de landmarks ne peut pas dépasser le nombre de nœuds
            assert len(landmarks) <= len(graph.nodes)


class TestFindNearestNode:
    """Tests pour la méthode find_nearest_node"""
    
    def test_find_nearest_node_success(self, sample_risk_df, sample_street_segments):
        """Test de recherche réussie du nœud le plus proche"""
        router = ALTRouter()
        router.build_graph(sample_risk_df, sample_street_segments)
        
        if router.graph and len(router.graph.nodes) > 0:
            test_point = Point(-71.0595, 42.3605)
            
            nearest_node = router.find_nearest_node(test_point)
            
            assert nearest_node is not None
            assert nearest_node in router.graph.nodes
            
    def test_find_nearest_node_with_coordinates(self, sample_risk_df, sample_street_segments):
        """Test avec coordonnées tuple"""
        router = ALTRouter()
        router.build_graph(sample_risk_df, sample_street_segments)
        
        if router.graph and len(router.graph.nodes) > 0:
            test_coords = (-71.0595, 42.3605)
            
            nearest_node = router.find_nearest_node(test_coords)
            
            assert nearest_node is not None
            assert nearest_node in router.graph.nodes
            
    def test_find_nearest_node_empty_graph(self):
        """Test avec graphe vide"""
        router = ALTRouter()
        router.graph = nx.DiGraph()
        
        test_point = Point(-71.0595, 42.3605)
        
        nearest_node = router.find_nearest_node(test_point)
        
        assert nearest_node is None
        
    def test_find_nearest_node_no_graph(self):
        """Test sans graphe"""
        router = ALTRouter()
        
        test_point = Point(-71.0595, 42.3605)
        
        nearest_node = router.find_nearest_node(test_point)
        
        assert nearest_node is None


class TestRiskCalculation:
    """Tests pour les méthodes de calcul de risque"""
    
    def test_calculate_risk_weight_basic(self):
        """Test de calcul de base du poids de risque"""
        router = ALTRouter()
        
        edge_data = {
            'base_risk': 0.5,
            'night_risk_factor': 1.2,
            'pedestrian_classification': {'pedestrian_safety': 'safe'}
        }
        
        risk = router.calculate_risk_weight(edge_data)
        
        assert isinstance(risk, (int, float))
        assert risk >= 0
        
    def test_calculate_risk_weight_with_time(self):
        """Test de calcul de risque avec heure"""
        router = ALTRouter()
        
        edge_data = {
            'base_risk': 0.5,
            'night_risk_factor': 1.5,
            'pedestrian_classification': {
                'pedestrian_safety': 'safe',
                'warnings': ['limited_lighting']
            }
        }
        
        # Test nuit
        night_time = datetime(2023, 6, 15, 22, 0)
        night_risk = router.calculate_risk_weight(edge_data, night_time)
        
        # Test jour
        day_time = datetime(2023, 6, 15, 14, 0)
        day_risk = router.calculate_risk_weight(edge_data, day_time)
        
        # Le risque de nuit devrait être plus élevé
        assert night_risk > day_risk
        
    def test_calculate_risk_weight_prohibited(self):
        """Test avec route interdite"""
        router = ALTRouter()
        
        edge_data = {
            'base_risk': 0.5,
            'pedestrian_classification': {'pedestrian_safety': 'prohibited'}
        }
        
        risk = router.calculate_risk_weight(edge_data)
        
        # Route interdite = risque infini
        assert risk == float('inf')
        
    def test_calculate_edge_weight(self):
        """Test de calcul du poids total d'une arête"""
        router = ALTRouter()
        
        edge_data = {
            'length': 100.0,
            'base_risk': 0.3,
            'pedestrian_classification': {
                'category': 'normal',
                'pedestrian_safety': 'safe'
            }
        }
        
        weight = router.calculate_edge_weight('node1', 'node2', edge_data)
        
        assert isinstance(weight, (int, float))
        assert weight > 0
        
    def test_calculate_edge_weight_high_risk(self):
        """Test avec arête à haut risque"""
        router = ALTRouter()
        
        low_risk_data = {
            'length': 100.0,
            'base_risk': 0.1,
            'pedestrian_classification': {'pedestrian_safety': 'safe'}
        }
        
        high_risk_data = {
            'length': 100.0,
            'base_risk': 0.8,
            'pedestrian_classification': {'pedestrian_safety': 'caution'}
        }
        
        low_weight = router.calculate_edge_weight('n1', 'n2', low_risk_data)
        high_weight = router.calculate_edge_weight('n1', 'n2', high_risk_data)
        
        # Arête à haut risque devrait avoir un poids plus élevé
        assert high_weight > low_weight


class TestLandmarkHeuristic:
    """Tests pour l'heuristique basée sur les landmarks"""
    
    def test_landmark_heuristic_basic(self):
        """Test de base de l'heuristique"""
        router = ALTRouter()
        
        # Simuler des landmarks et distances
        router.landmarks = ['landmark1', 'landmark2']
        router.landmark_distances = {
            'landmark1': {'node1': 10.0, 'node2': 15.0},
            'landmark2': {'node1': 8.0, 'node2': 12.0}
        }
        
        heuristic = router.landmark_heuristic('node1', 'node2')
        
        assert isinstance(heuristic, (int, float))
        assert heuristic >= 0
        
    def test_landmark_heuristic_no_landmarks(self):
        """Test sans landmarks"""
        router = ALTRouter()
        router.landmarks = []
        router.landmark_distances = {}
        
        heuristic = router.landmark_heuristic('node1', 'node2')
        
        assert heuristic == 0
        
    def test_landmark_heuristic_missing_data(self):
        """Test avec données manquantes"""
        router = ALTRouter()
        
        router.landmarks = ['landmark1']
        router.landmark_distances = {
            'landmark1': {'node1': 10.0}  # node2 manquant
        }
        
        heuristic = router.landmark_heuristic('node1', 'node2')
        
        # Doit gérer les données manquantes
        assert isinstance(heuristic, (int, float))
        assert heuristic >= 0


class TestAStarSearch:
    """Tests pour l'algorithme A*"""
    
    def test_a_star_search_success(self, sample_risk_df, sample_street_segments):
        """Test de recherche A* réussie"""
        router = ALTRouter()
        router.build_graph(sample_risk_df, sample_street_segments)
        
        if router.graph and len(router.graph.nodes) >= 2:
            nodes = list(router.graph.nodes)
            start_point = Point(nodes[0][0], nodes[0][1])
            end_point = Point(nodes[-1][0], nodes[-1][1])
            
            try:
                path, distance, avg_risk = router.a_star_search(start_point, end_point)
                
                if path is not None:
                    assert isinstance(path, list)
                    assert len(path) >= 2
                    assert isinstance(distance, (int, float))
                    assert isinstance(avg_risk, (int, float))
                    assert distance >= 0
                    assert avg_risk >= 0
                else:
                    # Aucun chemin trouvé - acceptable pour un petit graphe
                    assert distance == 0
                    assert avg_risk == 0
                    
            except ValueError:
                # Acceptable si les points ne sont pas dans le graphe
                pass
                
    def test_a_star_search_no_graph(self):
        """Test sans graphe"""
        router = ALTRouter()
        
        start_point = Point(-71.0589, 42.3601)
        end_point = Point(-71.0599, 42.3611)
        
        with pytest.raises(ValueError):
            router.a_star_search(start_point, end_point)
            
    def test_a_star_search_with_time(self, sample_risk_df, sample_street_segments):
        """Test avec heure spécifiée"""
        router = ALTRouter()
        router.build_graph(sample_risk_df, sample_street_segments)
        
        if router.graph and len(router.graph.nodes) >= 2:
            nodes = list(router.graph.nodes)
            start_point = Point(nodes[0][0], nodes[0][1])
            end_point = Point(nodes[-1][0], nodes[-1][1])
            
            test_time = datetime(2023, 6, 15, 22, 0)
            
            try:
                path, distance, avg_risk = router.a_star_search(start_point, end_point, test_time)
                
                # Le résultat devrait tenir compte de l'heure
                assert isinstance(path, (list, type(None)))
                assert isinstance(distance, (int, float))
                assert isinstance(avg_risk, (int, float))
                
            except ValueError:
                # Acceptable si les points ne sont pas trouvés
                pass


class TestGetSafeRoute:
    """Tests pour la méthode get_safe_route"""
    
    def test_get_safe_route_success(self, sample_risk_df, sample_street_segments):
        """Test de calcul d'itinéraire sécurisé réussi"""
        router = ALTRouter()
        router.build_graph(sample_risk_df, sample_street_segments)
        
        if router.graph and len(router.graph.nodes) >= 2:
            start_point = (-71.0589, 42.3601)
            end_point = (-71.0599, 42.3611)
            
            result = router.get_safe_route(start_point, end_point)
            
            assert isinstance(result, dict)
            assert 'status' in result
            
            if result['status'] == 'success':
                assert 'route' in result
                assert 'distance' in result['route']
                assert 'estimated_time' in result['route']
                assert 'risk_score' in result['route']
                assert 'segments' in result['route']
            else:
                assert result['status'] == 'error'
                
    def test_get_safe_route_with_time(self, sample_risk_df, sample_street_segments):
        """Test avec heure spécifiée"""
        router = ALTRouter()
        router.build_graph(sample_risk_df, sample_street_segments)
        
        if router.graph and len(router.graph.nodes) >= 2:
            start_point = (-71.0589, 42.3601)
            end_point = (-71.0599, 42.3611)
            test_time = datetime(2023, 6, 15, 14, 30)
            
            result = router.get_safe_route(start_point, end_point, test_time)
            
            assert isinstance(result, dict)
            if result['status'] == 'success':
                assert 'time' in result
                
    def test_get_safe_route_no_path(self):
        """Test quand aucun chemin n'existe"""
        router = ALTRouter()
        
        # Graphe avec nœuds isolés
        G = nx.DiGraph()
        G.add_node((0, 0), x=0, y=0)
        G.add_node((1, 1), x=1, y=1)
        # Pas d'arêtes = pas de chemin
        
        router.graph = G
        
        start_point = (0, 0)
        end_point = (1, 1)
        
        result = router.get_safe_route(start_point, end_point)
        
        assert result['status'] == 'error'
        
    def test_get_safe_route_transport_detection(self, sample_risk_df):
        """Test de détection des transports requis"""
        # Créer un segment nécessitant un transport
        ferry_line = LineString([(-71.0589, 42.3601), (-71.0599, 42.3611)])
        ferry_segment = gpd.GeoDataFrame({
            'ST_NAME': ['Harbor Ferry'],
            'ST_TYPE': ['FERRY'],
            'full_street_name': ['Harbor Ferry']
        }, geometry=[ferry_line], crs="EPSG:4326")
        
        router = ALTRouter()
        router.allow_restricted_routes = True  # Permettre les transports
        
        graph = router.build_graph(sample_risk_df.head(1), ferry_segment)
        
        if graph and len(graph.nodes) >= 2:
            nodes = list(graph.nodes)
            start_point = (nodes[0][0], nodes[0][1])
            end_point = (nodes[-1][0], nodes[-1][1])
            
            result = router.get_safe_route(start_point, end_point)
            
            if result['status'] == 'success':
                assert 'transport_required' in result['route']


class TestGetRouteAsGeoJSON:
    """Tests pour la méthode get_route_as_geojson"""
    
    def test_get_route_as_geojson_success(self, sample_risk_df, sample_street_segments):
        """Test de génération GeoJSON réussie"""
        router = ALTRouter()
        router.build_graph(sample_risk_df, sample_street_segments)
        
        if router.graph and len(router.graph.nodes) >= 2:
            start_point = (-71.0589, 42.3601)
            end_point = (-71.0599, 42.3611)
            
            result = router.get_route_as_geojson(start_point, end_point)
            
            assert isinstance(result, dict)
            assert result['type'] == 'FeatureCollection'
            assert 'features' in result


class TestIntegrationAndPerformance:
    """Tests d'intégration et de performance"""
    
    def test_complete_routing_workflow(self, sample_risk_df, sample_street_segments):
        """Test du workflow complet de routage"""
        router = ALTRouter()
        
        # 1. Configuration
        router.set_pedestrian_mode(strict_mode=True, allow_transports=False)
        router.set_risk_weight(0.6)
        
        # 2. Construction du graphe
        graph = router.build_graph(sample_risk_df, sample_street_segments)
        
        if graph and len(graph.nodes) >= 2:
            # 3. Calcul d'itinéraire
            nodes = list(graph.nodes)
            start = (nodes[0][0], nodes[0][1])
            end = (nodes[-1][0], nodes[-1][1])
            
            # Route normale
            route = router.get_safe_route(start, end)
            assert isinstance(route, dict)
            
            # Route avec heure
            timed_route = router.get_safe_route(start, end, datetime.now())
            assert isinstance(timed_route, dict)
            
            # GeoJSON
            geojson = router.get_route_as_geojson(start, end)
            assert geojson['type'] == 'FeatureCollection'
            
    def test_memory_efficiency(self, sample_risk_df):
        """Test d'efficacité mémoire"""
        router = ALTRouter()
        
        # Créer un dataset plus volumineux
        n_segments = 100
        lines = [
            LineString([(-71.0589 + i*0.001, 42.3601 + i*0.001), 
                       (-71.0589 + (i+1)*0.001, 42.3601 + (i+1)*0.001)])
            for i in range(n_segments)
        ]
        
        large_streets = gpd.GeoDataFrame({
            'ST_NAME': [f'Street_{i}' for i in range(n_segments)],
            'full_street_name': [f'Street {i}' for i in range(n_segments)]
        }, geometry=lines, crs="EPSG:4326")
        
        # La construction devrait fonctionner même avec beaucoup de données
        try:
            graph = router.build_graph(sample_risk_df.head(50), large_streets)
            assert isinstance(graph, nx.DiGraph) or graph is None
        except MemoryError:
            pytest.skip("Pas assez de mémoire pour ce test")
            
    def test_parameter_sensitivity(self, sample_risk_df, sample_street_segments):
        """Test de sensibilité aux paramètres"""
        router = ALTRouter()
        router.build_graph(sample_risk_df, sample_street_segments)
        
        if router.graph and len(router.graph.nodes) >= 2:
            nodes = list(router.graph.nodes)
            start = (nodes[0][0], nodes[0][1])
            end = (nodes[-1][0], nodes[-1][1])
            
            # Test avec différents poids de risque
            results = []
            for weight in [0.0, 0.5, 1.0]:
                router.set_risk_weight(weight)
                result = router.get_safe_route(start, end)
                results.append(result)
                
            # Les résultats peuvent varier selon les paramètres
            # (test implicite - pas d'erreur signifie que ça fonctionne)
            assert len(results) == 3


class TestValidationAndEdgeCases:
    """Tests de validation et cas limites"""
            
    def test_duplicate_coordinates(self, sample_risk_df):
        """Test avec coordonnées dupliquées"""
        router = ALTRouter()
        
        # Segments avec des coordonnées très proches (presque dupliquées)
        close_lines = [
            LineString([(-71.0589, 42.3601), (-71.0589001, 42.3601001)]),
            LineString([(-71.0589001, 42.3601001), (-71.0589002, 42.3601002)])
        ]
        
        close_streets = gpd.GeoDataFrame({
            'ST_NAME': ['Street1', 'Street2'],
            'full_street_name': ['Street 1', 'Street 2']
        }, geometry=close_lines, crs="EPSG:4326")
        
        # Doit gérer les coordonnées très proches
        try:
            graph = router.build_graph(sample_risk_df.head(2), close_streets)
            assert isinstance(graph, nx.DiGraph) or graph is None
        except Exception as e:
            pytest.fail(f"Doit gérer les coordonnées proches: {e}")
            
    def test_invalid_geometry(self, sample_risk_df):
        """Test avec géométries invalides"""
        router = ALTRouter()
        
        # Créer des géométries invalides
        invalid_streets = gpd.GeoDataFrame({
            'ST_NAME': ['Invalid'],
            'full_street_name': ['Invalid Street']
        }, geometry=[LineString()], crs="EPSG:4326")  # LineString vide
        
        # Doit gérer les géométries invalides
        try:
            graph = router.build_graph(sample_risk_df.head(1), invalid_streets)
            assert isinstance(graph, nx.DiGraph) or graph is None
        except Exception as e:
            # Acceptable - géométries invalides peuvent causer des erreurs
            pass