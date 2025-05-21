from shapely.geometry import Point, LineString
import geopandas as gpd

def find_nearest_entity(point, entities_gdf, max_distance=100):
    """
    Trouve l'entité (segment de rue ou intersection) la plus proche d'un point donné
    
    Args:
        point (Point): Point géographique
        entities_gdf (GeoDataFrame): GeoDataFrame contenant les entités spatiales
        max_distance (float): Distance maximale de recherche en mètres
    
    Returns:
        int or None: Index de l'entité la plus proche ou None si aucune entité trouvée
    """
    # Assurer que les données ont un CRS défini
    if not entities_gdf.crs:
        entities_gdf.set_crs("EPSG:4326", inplace=True)
    
    # Créer un GeoDataFrame temporaire pour le point
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    
    # Convertir en système de coordonnées projeté adapté à Boston (NAD83 / Massachusetts Mainland)
    entities_projected = entities_gdf.to_crs("EPSG:26986")
    point_projected = point_gdf.to_crs("EPSG:26986")
    
    # Calculer la distance de chaque entité au point (en mètres)
    distances = entities_projected.geometry.distance(point_projected.geometry[0])
    
    # Trouver l'index de l'entité la plus proche
    if len(distances) > 0:
        min_distance_idx = distances.idxmin()
        min_distance = distances[min_distance_idx]
        
        # Vérifier si la distance est inférieure à la distance maximale (en mètres)
        if min_distance <= max_distance:
            return min_distance_idx
    
    return None

def identify_intersections(streets_gdf, buffer_distance=5):
    """
    Identifie les intersections à partir des segments de rue
    
    Args:
        streets_gdf (GeoDataFrame): GeoDataFrame des segments de rue
        buffer_distance (float): Distance tampon en mètres pour regrouper les points d'intersection proches
    
    Returns:
        GeoDataFrame: GeoDataFrame des intersections identifiées
    """
    if streets_gdf.empty:
        return gpd.GeoDataFrame(crs="EPSG:4326")
    
    # Projeter en système de coordonnées local pour des calculs précis
    streets_projected = streets_gdf.to_crs("EPSG:26986")  # NAD83 / Massachusetts Mainland
    
    # Extraire les points de début et de fin de chaque segment
    endpoints = []
    for _, row in streets_projected.iterrows():
        if isinstance(row.geometry, LineString):
            endpoints.append(Point(row.geometry.coords[0]))
            endpoints.append(Point(row.geometry.coords[-1]))
    
    # Trouver les points qui apparaissent plusieurs fois (intersections)
    from collections import Counter
    point_counter = Counter([(round(p.x, 1), round(p.y, 1)) for p in endpoints])
    intersection_points = [Point(x, y) for (x, y), count in point_counter.items() if count > 1]
    
    # Créer un GeoDataFrame des intersections
    if intersection_points:
        intersection_geoms = gpd.GeoSeries(intersection_points, crs="EPSG:26986")
        intersections_gdf = gpd.GeoDataFrame(geometry=intersection_geoms, crs="EPSG:26986")
        
        # Regrouper les intersections proches en utilisant un buffer et une dissolution
        # Cela évite d'avoir plusieurs points pour une même intersection
        buffered = intersections_gdf.buffer(buffer_distance)
        dissolved = buffered.unary_union
        
        # Extraire les centroïdes des polygones dissous
        if hasattr(dissolved, 'geoms'):
            # MultiPolygon
            centroids = [polygon.centroid for polygon in dissolved.geoms]
        else:
            # Polygon
            centroids = [dissolved.centroid]
        
        # Créer un nouveau GeoDataFrame avec les centroïdes
        refined_intersections = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(centroids, crs="EPSG:26986"),
            crs="EPSG:26986"
        )
        
        # Ajouter des attributs utiles
        refined_intersections['intersection_id'] = range(len(refined_intersections))
        
        # Reconvertir en WGS84
        refined_intersections = refined_intersections.to_crs("EPSG:4326")
        
        return refined_intersections
    
    return gpd.GeoDataFrame(crs="EPSG:4326")

def create_pedestrian_heatmap_data(risk_df):
    """
    Crée des données pour une carte de chaleur du risque piéton
    
    Args:
        risk_df (GeoDataFrame): GeoDataFrame avec les scores de risque
    
    Returns:
        list: Liste de points pondérés pour la carte de chaleur
    """
    # Filtrer les entités avec un score de risque
    risk_df = risk_df.dropna(subset=['risk_score'])
    
    # Créer une liste de points pondérés
    heatmap_data = []
    
    for idx, entity in risk_df.iterrows():
        # Obtenir le centroïde de l'entité
        centroid = entity.geometry.centroid
        
        # Ajouter plusieurs points proportionnellement au score de risque
        # Plus le score est élevé, plus on ajoute de points pour créer un "point chaud"
        weight = int(entity['risk_score'] * 10) + 1  # Convertir le score en nombre de points
        
        for _ in range(weight):
            heatmap_data.append([centroid.y, centroid.x])
    
    return heatmap_data

def create_geojson_from_risk_zones(risk_zones, include_recommendations=False):
    """
    Crée un GeoJSON à partir des informations sur les zones à risque pour les piétons
    
    Args:
        risk_zones (list): Liste de dictionnaires contenant les informations des zones à risque
        include_recommendations (bool): Si True, inclut les recommandations de sécurité
    
    Returns:
        dict: GeoJSON contenant les zones à risque
    """
    features = []
    
    for zone in risk_zones:
        properties = {
            "name": zone['name'],
            "type": zone['type'],
            "risk_score": zone['risk_score'],
            "risk_level": zone['risk_level'],
            "pedestrian_crashes": zone['ped_crash_count'],
            "night_crashes": zone.get('ped_crash_night_count', 0)
        }
        
        # Ajouter les recommandations si demandé
        if include_recommendations and 'recommendations' in zone:
            properties["problems"] = zone.get('problems_identified', [])
            properties["recommendations"] = zone['recommendations']
        
        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "Point",
                "coordinates": [zone['coordinates']['lon'], zone['coordinates']['lat']]
            }
        }
        
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson

def find_nearby_amenities(point, distance=500, amenity_types=None):
    """
    Trouve les équipements proches d'un point donné qui pourraient affecter la sécurité des piétons
    
    Args:
        point (Point): Point géographique
        distance (float): Distance de recherche en mètres
        amenity_types (list): Types d'équipements à rechercher
        
    Returns:
        dict: Dictionnaire contenant les équipements trouvés
    """
    # Par défaut, rechercher des équipements pertinents pour la sécurité des piétons
    if amenity_types is None:
        amenity_types = [
            'school', 'hospital', 'bus_stop', 'subway_entrance', 
            'crossing', 'traffic_signals', 'street_lamp'
        ]
    
    # Note: Cette fonction est une ébauche et nécessiterait une source de données OSM
    # Pour une implémentation complète, il faudrait utiliser l'API Overpass ou des données OSM préchargées
    
    # Simuler des résultats pour l'exemple
    return {
        "amenities_count": 0,
        "amenities_by_type": {},
        "message": "Fonctionnalité de recherche d'équipements non implémentée"
    }