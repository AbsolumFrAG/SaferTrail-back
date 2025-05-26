import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np

from src.geo_utils import (
    find_nearest_entity,
    identify_intersections,
    create_pedestrian_heatmap_data,
    create_geojson_from_risk_zones,
    create_geojson_from_street_segments,
    find_nearby_amenities
)


class TestFindNearestEntity:
    """Tests pour la fonction find_nearest_entity"""
    
    def test_find_nearest_entity_success(self, sample_street_segments):
        """Test de recherche d'entité la plus proche - cas de succès"""
        test_point = Point(-71.0595, 42.3605)
        
        nearest_idx = find_nearest_entity(test_point, sample_street_segments, max_distance=1000)
        
        assert nearest_idx is not None
        assert isinstance(nearest_idx, (int, np.integer))
        assert nearest_idx in sample_street_segments.index
    
    def test_find_nearest_entity_too_far(self, sample_street_segments):
        """Test quand aucune entité n'est assez proche"""
        # Point très éloigné
        test_point = Point(-70.0000, 41.0000)
        
        nearest_idx = find_nearest_entity(test_point, sample_street_segments, max_distance=100)
        
        assert nearest_idx is None
    
    def test_find_nearest_entity_empty_gdf(self):
        """Test avec un GeoDataFrame vide"""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        test_point = Point(-71.0595, 42.3605)
        
        nearest_idx = find_nearest_entity(test_point, empty_gdf)
        
        assert nearest_idx is None
    
    def test_find_nearest_entity_no_crs(self, sample_street_segments):
        """Test quand le GeoDataFrame n'a pas de CRS défini"""
        # Supprimer le CRS
        sample_street_segments.crs = None
        test_point = Point(-71.0595, 42.3605)
        
        nearest_idx = find_nearest_entity(test_point, sample_street_segments)
        
        # Doit fonctionner même sans CRS initial
        assert nearest_idx is not None


class TestIdentifyIntersections:
    """Tests pour la fonction identify_intersections"""
    
    def test_identify_intersections_success(self, sample_street_segments):
        """Test d'identification d'intersections - cas de succès"""
        # Modifier les segments pour créer une intersection
        lines = [
            LineString([(-71.0600, 42.3600), (-71.0590, 42.3610)]),  # Segment 1
            LineString([(-71.0595, 42.3595), (-71.0595, 42.3615)]),  # Segment 2 qui croise le 1
            LineString([(-71.0590, 42.3610), (-71.0580, 42.3620)])   # Segment 3 connecté au 1
        ]
        
        gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
        
        intersections = identify_intersections(gdf, buffer_distance=10)
        
        assert isinstance(intersections, gpd.GeoDataFrame)
        assert intersections.crs.to_string() == "EPSG:4326"
        assert len(intersections) >= 1  # Au moins une intersection détectée
        
        if len(intersections) > 0:
            assert 'intersection_id' in intersections.columns
            assert all(isinstance(geom, Point) for geom in intersections.geometry)


class TestCreatePedestrianHeatmapData:
    """Tests pour la fonction create_pedestrian_heatmap_data"""
    
    def test_create_heatmap_data_success(self, sample_risk_df):
        """Test de création de données de heatmap - cas de succès"""
        heatmap_data = create_pedestrian_heatmap_data(sample_risk_df)
        
        assert isinstance(heatmap_data, list)
        assert len(heatmap_data) > 0
        
        # Vérifier que chaque point a lat/lon
        for point in heatmap_data:
            assert len(point) == 2
            assert isinstance(point[0], (int, float))  # lat
            assert isinstance(point[1], (int, float))  # lon
    
    def test_create_heatmap_data_empty_gdf(self):
        """Test avec un GeoDataFrame vide"""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty_gdf['risk_score'] = []
        
        heatmap_data = create_pedestrian_heatmap_data(empty_gdf)
        
        assert isinstance(heatmap_data, list)
        assert len(heatmap_data) == 0


class TestCreateGeoJsonFromRiskZones:
    """Tests pour la fonction create_geojson_from_risk_zones"""
    
    def test_create_geojson_basic(self):
        """Test de création de GeoJSON basique"""
        risk_zones = [
            {
                'name': 'Test Zone 1',
                'type': 'intersection',
                'risk_score': 0.8,
                'risk_level': 'Élevé',
                'ped_crash_count': 3,
                'ped_crash_night_count': 1,
                'coordinates': {'lat': 42.3601, 'lon': -71.0589}
            },
            {
                'name': 'Test Zone 2',
                'type': 'street_segment',
                'risk_score': 0.6,
                'risk_level': 'Modéré',
                'ped_crash_count': 1,
                'ped_crash_night_count': 0,
                'coordinates': {'lat': 42.3611, 'lon': -71.0599}
            }
        ]
        
        geojson = create_geojson_from_risk_zones(risk_zones)
        
        assert isinstance(geojson, dict)
        assert geojson['type'] == 'FeatureCollection'
        assert 'features' in geojson
        assert len(geojson['features']) == 2
        
        # Vérifier la première feature
        feature1 = geojson['features'][0]
        assert feature1['type'] == 'Feature'
        assert feature1['geometry']['type'] == 'Point'
        assert feature1['geometry']['coordinates'] == [-71.0589, 42.3601]
        assert feature1['properties']['name'] == 'Test Zone 1'
        assert feature1['properties']['risk_score'] == 0.8
    
    def test_create_geojson_with_recommendations(self):
        """Test avec recommandations incluses"""
        risk_zones = [
            {
                'name': 'Test Zone',
                'type': 'intersection',
                'risk_score': 0.9,
                'risk_level': 'Très élevé',
                'ped_crash_count': 5,
                'coordinates': {'lat': 42.3601, 'lon': -71.0589},
                'problems_identified': ['Accidents fréquents', 'Mauvais éclairage'],
                'recommendations': ['Installer des feux', 'Améliorer l\'éclairage']
            }
        ]
        
        geojson = create_geojson_from_risk_zones(risk_zones, include_recommendations=True)
        
        feature = geojson['features'][0]
        assert 'problems' in feature['properties']
        assert 'recommendations' in feature['properties']
        assert len(feature['properties']['recommendations']) == 2
    
    def test_create_geojson_empty_list(self):
        """Test avec une liste vide"""
        geojson = create_geojson_from_risk_zones([])
        
        assert isinstance(geojson, dict)
        assert geojson['type'] == 'FeatureCollection'
        assert geojson['features'] == []


class TestCreateGeoJsonFromStreetSegments:
    """Tests pour la fonction create_geojson_from_street_segments"""
    
    def test_create_geojson_street_segments_basic(self, sample_risk_df):
        """Test de création de GeoJSON pour segments de rue"""
        geojson = create_geojson_from_street_segments(sample_risk_df)
        
        assert isinstance(geojson, dict)
        assert geojson['type'] == 'FeatureCollection'
        assert 'features' in geojson
        
        # Doit inclure tous les segments de rue
        street_segments = sample_risk_df[sample_risk_df['entity_type'] == 'street_segment']
        assert len(geojson['features']) == len(street_segments)
        
        # Vérifier une feature
        if geojson['features']:
            feature = geojson['features'][0]
            assert feature['type'] == 'Feature'
            assert feature['geometry']['type'] == 'LineString'
            assert 'name' in feature['properties']
            assert 'risk_score' in feature['properties']
    
    def test_create_geojson_with_min_risk_filter(self, sample_risk_df):
        """Test avec filtrage par risque minimum"""
        min_risk = 0.5
        geojson = create_geojson_from_street_segments(sample_risk_df, min_risk=min_risk)
        
        # Vérifier que tous les segments retournés ont un risque >= min_risk
        for feature in geojson['features']:
            assert feature['properties']['risk_score'] >= min_risk
    
    def test_create_geojson_with_limit(self, sample_risk_df):
        """Test avec limitation du nombre de segments"""
        limit = 2
        geojson = create_geojson_from_street_segments(sample_risk_df, limit=limit)
        
        assert len(geojson['features']) <= limit


class TestFindNearbyAmenities:
    """Tests pour la fonction find_nearby_amenities"""
    
    def test_find_nearby_amenities_basic(self):
        """Test de base pour la recherche d'équipements à proximité"""
        test_point = Point(-71.0589, 42.3601)
        
        result = find_nearby_amenities(test_point)
        
        # Fonction non implémentée, doit retourner un message
        assert isinstance(result, dict)
        assert 'amenities_count' in result
        assert 'message' in result
        assert "non implémentée" in result['message']
    
    def test_find_nearby_amenities_with_types(self):
        """Test avec types d'équipements spécifiés"""
        test_point = Point(-71.0589, 42.3601)
        amenity_types = ['school', 'hospital']
        
        result = find_nearby_amenities(test_point, amenity_types=amenity_types)
        
        assert isinstance(result, dict)
        assert 'amenities_count' in result
    
    def test_find_nearby_amenities_custom_distance(self):
        """Test avec distance personnalisée"""
        test_point = Point(-71.0589, 42.3601)
        
        result = find_nearby_amenities(test_point, distance=1000)
        
        assert isinstance(result, dict)


class TestGeoUtilsIntegration:
    """Tests d'intégration pour les fonctions geo_utils"""
    
    def test_workflow_risk_zones_to_geojson(self, sample_risk_df):
        """Test du workflow complet : zones à risque -> GeoJSON"""
        # Simuler la sélection de zones à haut risque
        high_risk_zones = []
        
        for idx, zone in sample_risk_df[sample_risk_df['risk_score'] >= 0.7].iterrows():
            centroid = zone.geometry.centroid
            zone_info = {
                'name': zone.get('full_street_name', 'Zone inconnue'),
                'type': zone.get('entity_type', 'unknown'),
                'risk_score': float(zone.get('risk_score', 0)),
                'risk_level': str(zone.get('risk_level', 'Inconnu')),
                'ped_crash_count': int(zone.get('ped_crash_count', 0)),
                'ped_crash_night_count': int(zone.get('ped_crash_night_count', 0)),
                'coordinates': {'lat': centroid.y, 'lon': centroid.x}
            }
            high_risk_zones.append(zone_info)
        
        # Créer le GeoJSON
        geojson = create_geojson_from_risk_zones(high_risk_zones)
        
        assert len(geojson['features']) > 0
        assert all(f['properties']['risk_score'] >= 0.7 for f in geojson['features'])
    
    def test_heatmap_data_creation_workflow(self, sample_risk_df):
        """Test du workflow de création de données de heatmap"""
        # Créer les données de heatmap
        heatmap_data = create_pedestrian_heatmap_data(sample_risk_df)
        
        assert len(heatmap_data) > 0
        
        # Vérifier que les points correspondent aux zones à haut risque
        high_risk_count = len(sample_risk_df[sample_risk_df['risk_score'] > 0.5])
        # Il devrait y avoir plus de points de heatmap que de zones (pondération)
        assert len(heatmap_data) >= high_risk_count


# Tests de validation des données
class TestDataValidation:
    """Tests de validation des données géographiques"""
    
    def test_coordinate_validity(self):
        """Test de validation des coordonnées"""
        # Coordonnées valides pour Boston
        valid_point = Point(-71.0589, 42.3601)
        assert -72 < valid_point.x < -70  # Longitude Boston
        assert 42 < valid_point.y < 43    # Latitude Boston
    
    def test_geometry_validity(self, sample_street_segments):
        """Test de validité des géométries"""
        for geom in sample_street_segments.geometry:
            assert geom.is_valid
            assert isinstance(geom, LineString)
            assert len(geom.coords) >= 2
    
    def test_crs_consistency(self, sample_street_segments):
        """Test de cohérence du système de coordonnées"""
        assert sample_street_segments.crs is not None
        assert sample_street_segments.crs.to_string() == "EPSG:4326"