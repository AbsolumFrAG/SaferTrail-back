import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
from .geo_utils import identify_intersections

def create_spatial_dataframes(df_crashes, df_311, gdf_streets, df_crimes):
    """
    Convertit les DataFrames en GeoDataFrames avec géométrie
    
    Args:
        df_crashes (DataFrame): Données d'accidents
        df_311 (DataFrame): Données 311
        gdf_streets (GeoDataFrame): Données des segments de rue
        df_crimes (DataFrame): Données de crimes
    
    Returns:
        tuple: Tuple contenant les GeoDataFrames créés
    """
    print("Création des GeoDataFrames...")
    
    # Conversion des données d'accidents en GeoDataFrame
    if not df_crashes.empty:
        df_crashes_with_coords = df_crashes.dropna(subset=['lat', 'long'])
        geometry_crashes = [Point(xy) for xy in zip(df_crashes_with_coords['long'], df_crashes_with_coords['lat'])]
        gdf_crashes = gpd.GeoDataFrame(df_crashes_with_coords, geometry=geometry_crashes, crs="EPSG:4326")
    else:
        gdf_crashes = gpd.GeoDataFrame(crs="EPSG:4326")
    
    # Conversion des données 311 en GeoDataFrame
    if not df_311.empty:
        df_311_with_coords = df_311.dropna(subset=['latitude', 'longitude'])
        geometry_311 = [Point(xy) for xy in zip(df_311_with_coords['longitude'], df_311_with_coords['latitude'])]
        gdf_311 = gpd.GeoDataFrame(df_311_with_coords, geometry=geometry_311, crs="EPSG:4326")
    else:
        gdf_311 = gpd.GeoDataFrame(crs="EPSG:4326")
    
    # Conversion des données de crimes en GeoDataFrame
    if not df_crimes.empty:
        df_crimes_with_coords = df_crimes.dropna(subset=['Lat', 'Long'])
        geometry_crimes = [Point(xy) for xy in zip(df_crimes_with_coords['Long'], df_crimes_with_coords['Lat'])]
        gdf_crimes = gpd.GeoDataFrame(df_crimes_with_coords, geometry=geometry_crimes, crs="EPSG:4326")
    else:
        gdf_crimes = gpd.GeoDataFrame(crs="EPSG:4326")
    
    return gdf_crashes, gdf_311, gdf_crimes

def categorize_pedestrian_issues(gdf_311):
    """
    Catégorise les problèmes signalés dans les requêtes 311 liés aux piétons
    
    Args:
        gdf_311 (GeoDataFrame): GeoDataFrame des requêtes 311
    
    Returns:
        GeoDataFrame: GeoDataFrame avec les problèmes catégorisés
    """
    if gdf_311.empty:
        return gdf_311
    
    # Créer des colonnes pour les différentes catégories de problèmes
    gdf_311['sidewalk_issues'] = gdf_311['combined_text'].str.contains(
        'sidewalk|footpath|pavement', case=False, na=False).astype(int)
    
    gdf_311['crossing_issues'] = gdf_311['combined_text'].str.contains(
        'crosswalk|crossing|intersection', case=False, na=False).astype(int)
    
    gdf_311['lighting_issues'] = gdf_311['combined_text'].str.contains(
        'light|lighting|dark|lamp', case=False, na=False).astype(int)
    
    gdf_311['signal_issues'] = gdf_311['combined_text'].str.contains(
        'signal|traffic light|pedestrian signal', case=False, na=False).astype(int)
    
    return gdf_311

def prepare_pedestrian_risk_analysis(gdf_crashes, gdf_311, gdf_streets, gdf_crimes):
    """
    Prépare l'analyse de risque pour les piétons en identifiant les entités spatiales
    et en calculant les facteurs de risque
    
    Args:
        gdf_crashes (GeoDataFrame): GeoDataFrame des accidents de piétons
        gdf_311 (GeoDataFrame): GeoDataFrame des requêtes 311 liées aux piétons
        gdf_streets (GeoDataFrame): GeoDataFrame des segments de rue
        gdf_crimes (GeoDataFrame): GeoDataFrame des crimes liés à la sécurité des piétons
    
    Returns:
        GeoDataFrame: GeoDataFrame avec les entités spatiales et leurs scores de risque
    """
    print("Préparation de l'analyse de risque pour les piétons...")
    
    # Identifier les intersections (points critiques pour la sécurité des piétons)
    intersections_gdf = identify_intersections(gdf_streets)
    
    # Projeter tous les GeoDataFrames en projection locale pour des calculs précis
    crs_projected = "EPSG:26986"  # NAD83 / Massachusetts Mainland
    
    if not gdf_streets.empty:
        streets_projected = gdf_streets.to_crs(crs_projected)
    else:
        print("Aucun segment de rue disponible. Création d'un GeoDataFrame vide.")
        streets_projected = gpd.GeoDataFrame(crs=crs_projected)
        return streets_projected
    
    crashes_projected = gdf_crashes.to_crs(crs_projected) if not gdf_crashes.empty else gpd.GeoDataFrame(crs=crs_projected)
    reports_projected = gdf_311.to_crs(crs_projected) if not gdf_311.empty else gpd.GeoDataFrame(crs=crs_projected)
    crimes_projected = gdf_crimes.to_crs(crs_projected) if not gdf_crimes.empty else gpd.GeoDataFrame(crs=crs_projected)
    intersections_projected = intersections_gdf.to_crs(crs_projected) if not intersections_gdf.empty else gpd.GeoDataFrame(crs=crs_projected)
    
    # Initialiser un GeoDataFrame combiné avec les segments de rue et les intersections
    risk_df = streets_projected.copy()
    
    # Ajouter une colonne indiquant le type d'entité (segment ou intersection)
    risk_df['entity_type'] = 'street_segment'
    
    # Ajouter les intersections au GeoDataFrame combiné si elles existent
    if not intersections_projected.empty:
        intersections_projected['entity_type'] = 'intersection'
        # Fusionner les deux GeoDataFrames
        risk_df = pd.concat([risk_df, intersections_projected], ignore_index=True)
    
    # Initialiser des colonnes pour les facteurs de risque
    risk_df['ped_crash_count'] = 0
    risk_df['ped_crash_night_count'] = 0  # Accidents de nuit (visibilité réduite)
    risk_df['ped_311_count'] = 0
    risk_df['sidewalk_issues'] = 0
    risk_df['crossing_issues'] = 0
    risk_df['lighting_issues'] = 0
    risk_df['signal_issues'] = 0
    risk_df['crime_count'] = 0
    risk_df['buffer_distance'] = 50  # Distance tampon en mètres pour associer les incidents
    
    # Associer chaque accident de piéton à l'entité la plus proche
    if not crashes_projected.empty:
        print("Association des accidents de piétons...")
        for idx, crash in crashes_projected.iterrows():
            # Trouver l'entité (segment ou intersection) la plus proche
            distances = risk_df.geometry.distance(crash.geometry)
            nearest_idx = distances.idxmin()
            distance = distances[nearest_idx]
            
            # Si l'accident est à moins de X mètres, l'associer à l'entité
            buffer_distance = risk_df.loc[nearest_idx, 'buffer_distance']
            if distance <= buffer_distance:
                risk_df.at[nearest_idx, 'ped_crash_count'] += 1
                
                # Vérifier si c'est un accident de nuit (entre 18h et 6h)
                hour = crash.get('hour', -1)
                if hour >= 18 or hour < 6:
                    risk_df.at[nearest_idx, 'ped_crash_night_count'] += 1
    
    # Associer chaque signalement 311 à l'entité la plus proche
    if not reports_projected.empty:
        print("Association des signalements 311...")
        for idx, report in reports_projected.iterrows():
            # Trouver l'entité la plus proche
            distances = risk_df.geometry.distance(report.geometry)
            nearest_idx = distances.idxmin()
            distance = distances[nearest_idx]
            
            # Si le signalement est à moins de X mètres, l'associer à l'entité
            buffer_distance = risk_df.loc[nearest_idx, 'buffer_distance']
            if distance <= buffer_distance:
                risk_df.at[nearest_idx, 'ped_311_count'] += 1
                
                # Catégoriser le type de problème
                if hasattr(report, 'sidewalk_issues') and report.sidewalk_issues == 1:
                    risk_df.at[nearest_idx, 'sidewalk_issues'] += 1
                if hasattr(report, 'crossing_issues') and report.crossing_issues == 1:
                    risk_df.at[nearest_idx, 'crossing_issues'] += 1
                if hasattr(report, 'lighting_issues') and report.lighting_issues == 1:
                    risk_df.at[nearest_idx, 'lighting_issues'] += 1
                if hasattr(report, 'signal_issues') and report.signal_issues == 1:
                    risk_df.at[nearest_idx, 'signal_issues'] += 1
    
    # Associer chaque crime à l'entité la plus proche
    if not crimes_projected.empty:
        print("Association des crimes...")
        for idx, crime in crimes_projected.iterrows():
            # Trouver l'entité la plus proche
            distances = risk_df.geometry.distance(crime.geometry)
            nearest_idx = distances.idxmin()
            distance = distances[nearest_idx]
            
            # Si le crime est à moins de X mètres, l'associer à l'entité
            buffer_distance = risk_df.loc[nearest_idx, 'buffer_distance']
            if distance <= buffer_distance:
                risk_df.at[nearest_idx, 'crime_count'] += 1
    
    # Reconvertir en WGS84 pour la sortie
    risk_df = risk_df.to_crs("EPSG:4326")
    
    return risk_df

def calculate_pedestrian_risk_scores(risk_df):
    """
    Calcule les scores de risque pour les piétons
    
    Args:
        risk_df (GeoDataFrame): GeoDataFrame avec les facteurs de risque
    
    Returns:
        GeoDataFrame: GeoDataFrame avec les scores de risque calculés
    """
    print("Calcul des scores de risque pour les piétons...")
    
    # Normaliser les métriques de risque
    scaler = MinMaxScaler()
    risk_factors = [
        'ped_crash_count', 'ped_crash_night_count', 'ped_311_count',
        'sidewalk_issues', 'crossing_issues', 'lighting_issues', 'signal_issues', 'crime_count'
    ]
    
    # Copier le DataFrame pour éviter des avertissements
    df_temp = risk_df.copy()
    
    # Normaliser seulement si nous avons des données
    for col in risk_factors:
        if df_temp[col].sum() > 0:
            # Reshape est nécessaire pour MinMaxScaler
            values = df_temp[[col]].values
            normalized_values = scaler.fit_transform(values)
            df_temp[col + '_normalized'] = normalized_values
        else:
            df_temp[col + '_normalized'] = 0
    
    # Calculer un score de risque composite
    # Attribuer des poids différents à chaque facteur
    weights = {
        'ped_crash_count_normalized': 0.30,
        'ped_crash_night_count_normalized': 0.15,
        'ped_311_count_normalized': 0.10,
        'sidewalk_issues_normalized': 0.10,
        'crossing_issues_normalized': 0.15,
        'lighting_issues_normalized': 0.10,
        'signal_issues_normalized': 0.05,
        'crime_count_normalized': 0.05
    }
    
    # Calculer le score de risque pondéré
    df_temp['risk_score'] = 0
    for col, weight in weights.items():
        if col in df_temp.columns:
            df_temp['risk_score'] += df_temp[col] * weight
    
    # Appliquer un facteur de risque supplémentaire pour les intersections
    # Les intersections sont généralement plus dangereuses pour les piétons
    df_temp.loc[df_temp['entity_type'] == 'intersection', 'risk_score'] *= 1.3
    
    # Normaliser le score de risque final entre 0 et 1
    max_score = df_temp['risk_score'].max()
    if max_score > 0:
        df_temp['risk_score'] = df_temp['risk_score'] / max_score
    
    # Catégoriser le niveau de risque
    df_temp['risk_level'] = pd.cut(
        df_temp['risk_score'],
        bins=[0, 0.25, 0.5, 0.75, 1],
        labels=['Faible', 'Modéré', 'Élevé', 'Très élevé']
    )
    
    # Mettre à jour le GeoDataFrame original
    for col in df_temp.columns:
        if col not in risk_df.columns:
            risk_df[col] = df_temp[col]
    
    return risk_df

def identify_pedestrian_risk_clusters(risk_df, min_samples=2, eps=0.001):
    """
    Identifie les clusters de zones à risque pour les piétons
    
    Args:
        risk_df (GeoDataFrame): GeoDataFrame avec les scores de risque
        min_samples (int): Nombre minimum de points pour former un cluster
        eps (float): Distance maximale entre deux points pour être considérés comme voisins
        
    Returns:
        GeoDataFrame: GeoDataFrame avec les clusters identifiés
    """
    print("Identification des clusters de risque...")
    
    from sklearn.cluster import DBSCAN
    
    # Sélectionner les entités avec un score de risque élevé
    high_risk = risk_df[risk_df['risk_score'] > 0.5].copy()
    
    if len(high_risk) < min_samples:
        print("Pas assez d'entités à haut risque pour former des clusters.")
        return risk_df
    
    # Extraire les coordonnées des centroïdes
    high_risk['centroid_x'] = high_risk.geometry.centroid.x
    high_risk['centroid_y'] = high_risk.geometry.centroid.y
    
    # Préparer les données pour le clustering
    coords = high_risk[['centroid_x', 'centroid_y']].values
    
    # Appliquer DBSCAN pour identifier les clusters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    
    # Ajouter les étiquettes de cluster au DataFrame
    high_risk['cluster'] = dbscan.labels_
    
    # Initialiser la colonne cluster dans le DataFrame original
    if 'cluster' not in risk_df.columns:
        risk_df['cluster'] = -1  # -1 signifie pas de cluster
    
    # Mettre à jour le DataFrame de risque original
    for idx, row in high_risk.iterrows():
        if idx in risk_df.index and row['cluster'] != -1:  # -1 signifie pas de cluster
            risk_df.at[idx, 'cluster'] = row['cluster']
    
    # Compter le nombre de clusters identifiés (hors bruit)
    n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    print(f"Nombre de clusters identifiés: {n_clusters}")
    
    return risk_df

def process_data_for_pedestrian_safety(df_crashes, df_311, gdf_streets, df_crimes):
    """
    Traite les données pour l'analyse de la sécurité des piétons
    
    Args:
        df_crashes (DataFrame): Données d'accidents
        df_311 (DataFrame): Données 311
        gdf_streets (GeoDataFrame): Données des segments de rue
        df_crimes (DataFrame): Données de crimes
        
    Returns:
        GeoDataFrame: GeoDataFrame avec les scores de risque calculés
    """
    print("Traitement des données pour l'analyse de la sécurité des piétons...")
    
    # 1. Créer des GeoDataFrames
    gdf_crashes, gdf_311, gdf_crimes = create_spatial_dataframes(df_crashes, df_311, gdf_streets, df_crimes)
    
    # 2. Catégoriser les problèmes signalés dans les requêtes 311
    gdf_311 = categorize_pedestrian_issues(gdf_311)
    
    # 3. Préparer l'analyse de risque
    risk_df = prepare_pedestrian_risk_analysis(gdf_crashes, gdf_311, gdf_streets, gdf_crimes)
    
    # 4. Calculer les scores de risque
    risk_df = calculate_pedestrian_risk_scores(risk_df)
    
    # 5. Identifier les clusters de risque
    risk_df = identify_pedestrian_risk_clusters(risk_df)
    
    return risk_df