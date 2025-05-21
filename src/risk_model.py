import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from .time_based_risk_model import TimeBasedPedestrianRiskModel

class PedestrianRiskModel:
    """
    Modèle pour évaluer les risques pour les piétons à Boston en se basant sur
    les accidents impliquant des piétons, les problèmes d'infrastructure signalés
    et d'autres facteurs.
    """
    
    def __init__(self):
        """Initialise le modèle de risque pour les piétons"""
        self.risk_df = None
        self.streets_gdf = None
        self.intersections_gdf = None  # Stocker les intersections séparément
        self.model_path = Path(__file__).parent.parent / 'models' / 'pedestrian_risk_model.pkl'
        
        # Initialiser le modèle temporel
        self.time_model = TimeBasedPedestrianRiskModel()
    
    def load_data(self):
        """
        Charge les données nécessaires pour le modèle
        
        Returns:
            tuple: Tuple contenant les DataFrames chargés
        """
        from pathlib import Path
        
        data_dir = Path(__file__).parent.parent / 'data'
        
        # Chargement des données d'accidents
        print("Chargement des données d'accidents...")
        df_crashes = pd.read_csv(data_dir / 'vision-zero-crash-records.csv', encoding='utf-8')
        
        # Filtrer pour garder uniquement les accidents impliquant des piétons
        df_crashes['dispatch_ts'] = pd.to_datetime(df_crashes['dispatch_ts'], errors='coerce')
        df_crashes['year'] = df_crashes['dispatch_ts'].dt.year
        df_crashes['month'] = df_crashes['dispatch_ts'].dt.month
        df_crashes['hour'] = df_crashes['dispatch_ts'].dt.hour
        df_crashes['day_of_week'] = df_crashes['dispatch_ts'].dt.day_name()
        
        # Filtrer pour les 3 dernières années et les accidents impliquant des piétons
        three_years_ago = datetime.now() - timedelta(days=3*365)
        
        # Gérer les problèmes de timezone
        if hasattr(df_crashes['dispatch_ts'].dt, 'tz') and df_crashes['dispatch_ts'].dt.tz is not None:
            timezone_name = str(df_crashes['dispatch_ts'].dt.tz)
            print(f"Les dates ont un fuseau horaire: {timezone_name}")
            try:
                # Convertir la date de comparaison pour qu'elle ait le même fuseau horaire
                timezone = pytz.timezone(timezone_name.replace('tzfile(', '').replace(')', ''))
                three_years_ago = timezone.localize(three_years_ago)
            except Exception as e:
                print(f"Erreur lors de la conversion du fuseau horaire: {e}")
                # Si nous ne pouvons pas localiser, nous convertissons les dates en naïves
                df_crashes['dispatch_ts'] = df_crashes['dispatch_ts'].dt.tz_localize(None)
        
        df_ped_crashes = df_crashes[(df_crashes['dispatch_ts'] >= three_years_ago) & 
                                    (df_crashes['mode_type'] == 'ped')].copy()
        
        print(f"Nombre d'accidents impliquant des piétons: {len(df_ped_crashes)}")
        
        # Chargement des données 311
        print("Chargement des données 311...")
        df_311 = pd.read_csv(data_dir / '311.csv', encoding='utf-8', low_memory=False)
        
        # Filtrer pour les signalements liés à la sécurité des piétons
        pedestrian_keywords = [
            'Sidewalk', 'Crosswalk', 'Pedestrian', 'Walkway', 'Footpath',
            'Sidewalk Repair', 'Street Light', 'Traffic Signal'
        ]
        
        # Créer une colonne combinée pour la recherche de mots-clés
        columns_to_check = ['case_title', 'subject', 'reason', 'type']
        df_311['combined_text'] = ''
        for col in columns_to_check:
            if col in df_311.columns:
                df_311['combined_text'] += df_311[col].fillna('') + ' '
        
        # Filtrer les demandes liées aux piétons
        df_311['ped_related'] = df_311['combined_text'].str.contains(
            '|'.join(pedestrian_keywords), case=False, na=False
        )
        
        df_ped_311 = df_311[df_311['ped_related']].copy()
        df_ped_311['open_dt'] = pd.to_datetime(df_ped_311['open_dt'], errors='coerce')
        df_ped_311['year'] = df_ped_311['open_dt'].dt.year
        df_ped_311['month'] = df_ped_311['open_dt'].dt.month
        df_ped_311['hour'] = df_ped_311['open_dt'].dt.hour
        df_ped_311['day_of_week'] = df_ped_311['open_dt'].dt.day_name()
        
        # Gérer les problèmes de timezone pour 311
        if hasattr(df_ped_311['open_dt'].dt, 'tz') and df_ped_311['open_dt'].dt.tz is not None:
            timezone_name = str(df_ped_311['open_dt'].dt.tz)
            print(f"Les dates 311 ont un fuseau horaire: {timezone_name}")
            try:
                # Utiliser la même période de 3 ans déjà localisée si possible
                pass
            except:
                # Si différent fuseau horaire, convertir les dates en naïves
                df_ped_311['open_dt'] = df_ped_311['open_dt'].dt.tz_localize(None)
        
        # Filtrer pour les 3 dernières années
        # Soit convertir three_years_ago pour ce fuseau, soit convertir les dates en naïves
        # Pour simplifier, nous allons convertir les dates en naïves si nécessaire
        if hasattr(df_ped_311['open_dt'].dt, 'tz') and df_ped_311['open_dt'].dt.tz is not None:
            # Si les dates ont un fuseau horaire mais qu'il est différent de celui des accidents
            if not hasattr(three_years_ago, 'tzinfo') or three_years_ago.tzinfo is None:
                # Convertir les dates en naïves
                df_ped_311['open_dt'] = df_ped_311['open_dt'].dt.tz_localize(None)
                three_years_ago_311 = three_years_ago
            else:
                three_years_ago_311 = three_years_ago
        else:
            # Si les dates sont déjà naïves
            three_years_ago_311 = three_years_ago
            if hasattr(three_years_ago, 'tzinfo') and three_years_ago.tzinfo is not None:
                # Si la date de comparaison a un fuseau, la convertir en naïve
                three_years_ago_311 = three_years_ago.replace(tzinfo=None)
        
        df_ped_311 = df_ped_311[df_ped_311['open_dt'] >= three_years_ago_311]
        
        print(f"Nombre de signalements 311 liés aux piétons: {len(df_ped_311)}")
        
        # Chargement des segments de rue
        print("Chargement des segments de rue...")
        try:
            gdf_streets = gpd.read_file(data_dir / 'boston-street-segments.geojson')
            print(f"Nombre de segments de rue: {len(gdf_streets)}")
        except Exception as e:
            print(f"Erreur lors du chargement des segments de rue: {e}")
            gdf_streets = gpd.GeoDataFrame()
        
        # Chargement des données de crimes
        print("Chargement des données de crimes...")
        df_crimes = pd.read_csv(data_dir / 'crimes-incident-report.csv', encoding='utf-8')
        
        # Filtrer pour les crimes pouvant affecter la sécurité des piétons
        df_crimes['OCCURRED_ON_DATE'] = pd.to_datetime(df_crimes['OCCURRED_ON_DATE'], errors='coerce')
        df_crimes['year'] = df_crimes['OCCURRED_ON_DATE'].dt.year
        df_crimes['month'] = df_crimes['OCCURRED_ON_DATE'].dt.month
        df_crimes['hour'] = df_crimes['OCCURRED_ON_DATE'].dt.hour
        df_crimes['day_of_week'] = df_crimes['OCCURRED_ON_DATE'].dt.day_name()
        
        # Gérer les problèmes de timezone pour crimes
        if hasattr(df_crimes['OCCURRED_ON_DATE'].dt, 'tz') and df_crimes['OCCURRED_ON_DATE'].dt.tz is not None:
            timezone_name = str(df_crimes['OCCURRED_ON_DATE'].dt.tz)
            print(f"Les dates de crimes ont un fuseau horaire: {timezone_name}")
            try:
                # Utiliser la même période de 3 ans déjà localisée si possible
                pass
            except:
                # Si différent fuseau horaire, convertir les dates en naïves
                df_crimes['OCCURRED_ON_DATE'] = df_crimes['OCCURRED_ON_DATE'].dt.tz_localize(None)
        
        # Filtrer pour les 3 dernières années
        # Gérer les problèmes de timezone comme pour 311
        if hasattr(df_crimes['OCCURRED_ON_DATE'].dt, 'tz') and df_crimes['OCCURRED_ON_DATE'].dt.tz is not None:
            # Si les dates ont un fuseau horaire mais qu'il est différent de celui des accidents
            if not hasattr(three_years_ago, 'tzinfo') or three_years_ago.tzinfo is None:
                # Convertir les dates en naïves
                df_crimes['OCCURRED_ON_DATE'] = df_crimes['OCCURRED_ON_DATE'].dt.tz_localize(None)
                three_years_ago_crimes = three_years_ago
            else:
                three_years_ago_crimes = three_years_ago
        else:
            # Si les dates sont déjà naïves
            three_years_ago_crimes = three_years_ago
            if hasattr(three_years_ago, 'tzinfo') and three_years_ago.tzinfo is not None:
                # Si la date de comparaison a un fuseau, la convertir en naïve
                three_years_ago_crimes = three_years_ago.replace(tzinfo=None)
        
        df_crimes = df_crimes[df_crimes['OCCURRED_ON_DATE'] >= three_years_ago_crimes]
        
        # Crimes qui peuvent affecter la sécurité des piétons (agression, vol, etc.)
        ped_safety_keywords = [
            'ASSAULT', 'ROBBERY', 'LARCENY', 'HARASSMENT', 'THREATS', 
            'DISORDERLY', 'VANDALISM', 'PUBLIC DRINKING'
        ]
        
        df_crimes['ped_safety_related'] = df_crimes['OFFENSE_DESCRIPTION'].str.contains(
            '|'.join(ped_safety_keywords), case=False, na=False
        )
        
        df_ped_crimes = df_crimes[df_crimes['ped_safety_related']].copy()
        
        print(f"Nombre de crimes liés à la sécurité des piétons: {len(df_ped_crimes)}")
        
        return df_ped_crashes, df_ped_311, gdf_streets, df_ped_crimes
    
    def prepare_spatial_data(self, df_ped_crashes, df_ped_311, gdf_streets, df_ped_crimes):
        """
        Prépare les données spatiales pour l'analyse
        
        Args:
            df_ped_crashes (DataFrame): Données d'accidents impliquant des piétons
            df_ped_311 (DataFrame): Données 311 liées aux piétons
            gdf_streets (GeoDataFrame): Données des segments de rue
            df_ped_crimes (DataFrame): Données de crimes liés à la sécurité des piétons
            
        Returns:
            tuple: Tuple contenant les GeoDataFrames préparés
        """
        print("Préparation des données spatiales...")
        
        # Conversion des données d'accidents en GeoDataFrame
        if not df_ped_crashes.empty:
            # Vérifier si les coordonnées existent
            if 'lat' in df_ped_crashes.columns and 'long' in df_ped_crashes.columns:
                # Filtrer pour éliminer les valeurs NaN dans les coordonnées
                df_crashes_with_coords = df_ped_crashes.dropna(subset=['lat', 'long'])
                
                if not df_crashes_with_coords.empty:
                    # Créer les points géométriques
                    geometry_crashes = [Point(xy) for xy in zip(df_crashes_with_coords['long'], df_crashes_with_coords['lat'])]
                    gdf_ped_crashes = gpd.GeoDataFrame(df_crashes_with_coords, geometry=geometry_crashes, crs="EPSG:4326")
                else:
                    print("Aucune coordonnée valide dans les données d'accidents.")
                    gdf_ped_crashes = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            else:
                print("Colonnes de coordonnées manquantes dans les données d'accidents.")
                gdf_ped_crashes = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        else:
            print("Aucune donnée d'accident disponible.")
            gdf_ped_crashes = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Conversion des données 311 en GeoDataFrame
        if not df_ped_311.empty:
            # Vérifier si les coordonnées existent
            if 'latitude' in df_ped_311.columns and 'longitude' in df_ped_311.columns:
                # Filtrer pour éliminer les valeurs NaN dans les coordonnées
                df_311_with_coords = df_ped_311.dropna(subset=['latitude', 'longitude'])
                
                if not df_311_with_coords.empty:
                    # Créer les points géométriques
                    geometry_311 = [Point(xy) for xy in zip(df_311_with_coords['longitude'], df_311_with_coords['latitude'])]
                    gdf_ped_311 = gpd.GeoDataFrame(df_311_with_coords, geometry=geometry_311, crs="EPSG:4326")
                else:
                    print("Aucune coordonnée valide dans les données 311.")
                    gdf_ped_311 = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            else:
                print("Colonnes de coordonnées manquantes dans les données 311.")
                gdf_ped_311 = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        else:
            print("Aucune donnée 311 disponible.")
            gdf_ped_311 = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Conversion des données de crimes en GeoDataFrame
        if not df_ped_crimes.empty:
            # Vérifier si les coordonnées existent
            if 'Lat' in df_ped_crimes.columns and 'Long' in df_ped_crimes.columns:
                # Filtrer pour éliminer les valeurs NaN dans les coordonnées
                df_crimes_with_coords = df_ped_crimes.dropna(subset=['Lat', 'Long'])
                
                if not df_crimes_with_coords.empty:
                    # Créer les points géométriques
                    geometry_crimes = [Point(xy) for xy in zip(df_crimes_with_coords['Long'], df_crimes_with_coords['Lat'])]
                    gdf_ped_crimes = gpd.GeoDataFrame(df_crimes_with_coords, geometry=geometry_crimes, crs="EPSG:4326")
                else:
                    print("Aucune coordonnée valide dans les données criminelles.")
                    gdf_ped_crimes = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            else:
                print("Colonnes de coordonnées manquantes dans les données criminelles.")
                gdf_ped_crimes = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        else:
            print("Aucune donnée criminelle disponible.")
            gdf_ped_crimes = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Préparation des segments de rue
        if gdf_streets is None or gdf_streets.empty:
            print("Aucun segment de rue disponible. Création d'un GeoDataFrame vide.")
            gdf_streets = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        else:
            # Assurer que la projection est en WGS84 (EPSG:4326)
            if gdf_streets.crs != "EPSG:4326":
                gdf_streets = gdf_streets.to_crs("EPSG:4326")
            
            # Créer une colonne pour le nom complet de la rue
            if 'full_street_name' not in gdf_streets.columns:
                gdf_streets['full_street_name'] = gdf_streets.apply(
                    lambda x: ' '.join(filter(None, [
                        x.get('PRE_DIR', ''), 
                        x.get('ST_NAME', ''), 
                        x.get('ST_TYPE', ''), 
                        x.get('SUF_DIR', '')
                    ])),
                    axis=1
                )
        
        # Identifier les intersections (important pour la sécurité des piétons)
        self.identify_intersections(gdf_streets)
        
        return gdf_ped_crashes, gdf_ped_311, gdf_streets, gdf_ped_crimes
    
    def identify_intersections(self, gdf_streets):
        """
        Identifie les intersections à partir des segments de rue
        
        Args:
            gdf_streets (GeoDataFrame): GeoDataFrame des segments de rue
        """
        print("Identification des intersections...")
        
        if gdf_streets.empty:
            self.intersections_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            return
        
        # Projeter en système de coordonnées local pour des calculs précis
        streets_projected = gdf_streets.to_crs("EPSG:26986")  # NAD83 / Massachusetts Mainland
        
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
            self.intersections_gdf = gpd.GeoDataFrame(geometry=intersection_geoms, crs="EPSG:26986")
            
            # Reconvertir en WGS84
            self.intersections_gdf = self.intersections_gdf.to_crs("EPSG:4326")
            
            # Ajouter des attributs aux intersections
            self.intersections_gdf['intersection_id'] = range(len(self.intersections_gdf))
            
            print(f"Nombre d'intersections identifiées: {len(self.intersections_gdf)}")
        else:
            self.intersections_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            print("Aucune intersection identifiée.")
    
    def calculate_pedestrian_risk_factors(self, gdf_ped_crashes, gdf_ped_311, gdf_streets, gdf_ped_crimes):
        """
        Calcule les facteurs de risque pour les piétons
        
        Args:
            gdf_ped_crashes (GeoDataFrame): Données spatiales des accidents impliquant des piétons
            gdf_ped_311 (GeoDataFrame): Données spatiales des signalements 311 liés aux piétons
            gdf_streets (GeoDataFrame): Données spatiales des segments de rue
            gdf_ped_crimes (GeoDataFrame): Données spatiales des crimes liés à la sécurité des piétons
            
        Returns:
            GeoDataFrame: DataFrame avec les scores de risque calculés
        """
        print("Calcul des facteurs de risque pour les piétons...")
        
        # Vérifier que tous les inputs sont des GeoDataFrames avec une géométrie valide
        # Si ce n'est pas le cas, créer des GeoDataFrames vides
        
        # Vérifier gdf_ped_crashes
        if not isinstance(gdf_ped_crashes, gpd.GeoDataFrame) or 'geometry' not in gdf_ped_crashes.columns:
            print("Les données d'accidents ne sont pas un GeoDataFrame valide. Création d'un GeoDataFrame vide.")
            gdf_ped_crashes = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Vérifier gdf_ped_311
        if not isinstance(gdf_ped_311, gpd.GeoDataFrame) or 'geometry' not in gdf_ped_311.columns:
            print("Les données 311 ne sont pas un GeoDataFrame valide. Création d'un GeoDataFrame vide.")
            gdf_ped_311 = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Vérifier gdf_streets
        if not isinstance(gdf_streets, gpd.GeoDataFrame) or 'geometry' not in gdf_streets.columns:
            print("Les données des segments de rue ne sont pas un GeoDataFrame valide. Création d'un GeoDataFrame vide.")
            gdf_streets = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Vérifier gdf_ped_crimes
        if not isinstance(gdf_ped_crimes, gpd.GeoDataFrame) or 'geometry' not in gdf_ped_crimes.columns:
            print("Les données criminelles ne sont pas un GeoDataFrame valide. Création d'un GeoDataFrame vide.")
            gdf_ped_crimes = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Convertir tous les GeoDataFrames en projection locale pour des calculs précis
        crs_projected = "EPSG:26986"  # NAD83 / Massachusetts Mainland
        
        if not gdf_streets.empty:
            streets_projected = gdf_streets.to_crs(crs_projected)
        else:
            print("Aucun segment de rue disponible. Création d'un GeoDataFrame vide.")
            streets_projected = gpd.GeoDataFrame(geometry=[], crs=crs_projected)
            return streets_projected.to_crs("EPSG:4326")  # Retourner un GDF vide en WGS84
        
        crashes_projected = gdf_ped_crashes.to_crs(crs_projected) if not gdf_ped_crashes.empty else gpd.GeoDataFrame(geometry=[], crs=crs_projected)
        requests_projected = gdf_ped_311.to_crs(crs_projected) if not gdf_ped_311.empty else gpd.GeoDataFrame(geometry=[], crs=crs_projected)
        crimes_projected = gdf_ped_crimes.to_crs(crs_projected) if not gdf_ped_crimes.empty else gpd.GeoDataFrame(geometry=[], crs=crs_projected)
        
        # S'assurer que intersections_gdf est défini
        if not hasattr(self, 'intersections_gdf') or self.intersections_gdf is None:
            self.intersections_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        intersections_projected = self.intersections_gdf.to_crs(crs_projected) if not self.intersections_gdf.empty else gpd.GeoDataFrame(geometry=[], crs=crs_projected)
        
        # Initialiser un GeoDataFrame combiné avec les segments de rue et les intersections
        risk_df = streets_projected.copy()
        
        # Ajouter une colonne indiquant le type d'entité (segment ou intersection)
        risk_df['entity_type'] = 'street_segment'
        
        # Ajouter les intersections au GeoDataFrame combiné si elles existent
        if not intersections_projected.empty and 'geometry' in intersections_projected.columns:
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
        if not crashes_projected.empty and 'geometry' in crashes_projected.columns:
            print("Association des accidents de piétons...")
            for idx, crash in crashes_projected.iterrows():
                # S'assurer que la géométrie est valide
                if crash.geometry is None:
                    continue
                    
                # Trouver l'entité (segment ou intersection) la plus proche
                distances = risk_df.geometry.distance(crash.geometry)
                if distances.empty:
                    continue
                    
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
        if not requests_projected.empty and 'geometry' in requests_projected.columns:
            print("Association des signalements 311...")
            for idx, request in requests_projected.iterrows():
                # S'assurer que la géométrie est valide
                if request.geometry is None:
                    continue
                    
                # Trouver l'entité la plus proche
                distances = risk_df.geometry.distance(request.geometry)
                if distances.empty:
                    continue
                    
                nearest_idx = distances.idxmin()
                distance = distances[nearest_idx]
                
                # Si le signalement est à moins de X mètres, l'associer à l'entité
                buffer_distance = risk_df.loc[nearest_idx, 'buffer_distance']
                if distance <= buffer_distance:
                    risk_df.at[nearest_idx, 'ped_311_count'] += 1
                    
                    # Catégoriser le type de problème
                    request_text = str(request.get('combined_text', '')).lower()
                    if 'sidewalk' in request_text or 'footpath' in request_text:
                        risk_df.at[nearest_idx, 'sidewalk_issues'] += 1
                    if 'crosswalk' in request_text or 'crossing' in request_text:
                        risk_df.at[nearest_idx, 'crossing_issues'] += 1
                    if 'light' in request_text or 'lighting' in request_text or 'dark' in request_text:
                        risk_df.at[nearest_idx, 'lighting_issues'] += 1
                    if 'signal' in request_text or 'traffic light' in request_text:
                        risk_df.at[nearest_idx, 'signal_issues'] += 1
        
        # Associer chaque crime à l'entité la plus proche
        if not crimes_projected.empty and 'geometry' in crimes_projected.columns:
            print("Association des crimes...")
            for idx, crime in crimes_projected.iterrows():
                # S'assurer que la géométrie est valide
                if crime.geometry is None:
                    continue
                    
                # Trouver l'entité la plus proche
                distances = risk_df.geometry.distance(crime.geometry)
                if distances.empty:
                    continue
                    
                nearest_idx = distances.idxmin()
                distance = distances[nearest_idx]
                
                # Si le crime est à moins de X mètres, l'associer à l'entité
                buffer_distance = risk_df.loc[nearest_idx, 'buffer_distance']
                if distance <= buffer_distance:
                    risk_df.at[nearest_idx, 'crime_count'] += 1
        
        # Calculer les scores de risque
        self.calculate_risk_scores(risk_df)
        
        # Reconvertir en WGS84 pour la sortie
        risk_df = risk_df.to_crs("EPSG:4326")
        
        return risk_df
    
    def calculate_risk_scores(self, risk_df):
        """
        Calcule les scores de risque pour chaque entité
        
        Args:
            risk_df (GeoDataFrame): GeoDataFrame avec les facteurs de risque
        """
        print("Calcul des scores de risque...")
        
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
    
    def identify_risk_clusters(self, risk_df):
        """
        Identifie les clusters de zones à risque pour les piétons
        
        Args:
            risk_df (GeoDataFrame): GeoDataFrame avec les scores de risque
        
        Returns:
            GeoDataFrame: GeoDataFrame avec les clusters identifiés
        """
        print("Identification des clusters de risque...")
        
        # Sélectionner les entités avec un score de risque élevé
        high_risk = risk_df[risk_df['risk_score'] > 0.5].copy()
        
        if len(high_risk) < 3:
            print("Pas assez d'entités à haut risque pour former des clusters.")
            return risk_df
        
        # Extraire les coordonnées des centroïdes
        high_risk['centroid_x'] = high_risk.geometry.centroid.x
        high_risk['centroid_y'] = high_risk.geometry.centroid.y
        
        # Préparer les données pour le clustering
        coords = high_risk[['centroid_x', 'centroid_y']].values
        
        # Appliquer DBSCAN pour identifier les clusters
        # eps est la distance maximale entre deux points pour être considérés comme voisins (en degrés)
        # min_samples est le nombre minimum de points pour former un cluster
        dbscan = DBSCAN(eps=0.001, min_samples=2).fit(coords)
        
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
    
    def train(self, force_retrain=False):
        """
        Entraîne le modèle de risque pour les piétons
        
        Args:
            force_retrain (bool): Si True, force le réentraînement même si un modèle existe déjà
        
        Returns:
            bool: True si l'entraînement a réussi, False sinon
        """
        # Vérifier si un modèle existe déjà
        if not force_retrain and self.model_path.exists():
            try:
                self.load_model()
                print("Modèle chargé avec succès.")
                return True
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {e}")
        
        # Charger les données
        df_ped_crashes, df_ped_311, gdf_streets, df_ped_crimes = self.load_data()
        
        # Préparer les données spatiales
        gdf_ped_crashes, gdf_ped_311, gdf_streets, gdf_ped_crimes = self.prepare_spatial_data(
            df_ped_crashes, df_ped_311, gdf_streets, df_ped_crimes
        )
        
        # Calculer les facteurs de risque
        self.risk_df = self.calculate_pedestrian_risk_factors(
            gdf_ped_crashes, gdf_ped_311, gdf_streets, gdf_ped_crimes
        )
        
        # Identifier les clusters de risque
        self.identify_risk_clusters(self.risk_df)
        
        # Stocker les segments de rue
        self.streets_gdf = gdf_streets
        
        # Entraîner le modèle temporel
        self.train_time_model(df_ped_crashes, df_ped_311, df_ped_crimes)
        
        # Sauvegarder le modèle
        self.save_model()
        
        print("Modèle entraîné avec succès.")
        return True
    
    def train_time_model(self, df_crashes, df_311, df_crimes):
        """
        Entraîne le modèle temporel pour les prédictions horaires
        
        Args:
            df_crashes (DataFrame): Données d'accidents impliquant des piétons
            df_311 (DataFrame): Données 311 liées aux piétons
            df_crimes (DataFrame): Données de crimes liés à la sécurité des piétons
            
        Returns:
            dict: Résultats de l'analyse temporelle
        """
        print("Entraînement du modèle temporel...")
        
        # Analyser les patterns temporels
        time_analysis = self.time_model.analyze_temporal_patterns(
            df_crashes=df_crashes,
            df_311=df_311,
            df_crimes=df_crimes
        )
        
        # Sauvegarder le modèle temporel
        self.time_model.save_model()
        
        return time_analysis
    
    def save_model(self):
        """
        Sauvegarde le modèle entraîné
        
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            # Créer le répertoire s'il n'existe pas
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le modèle (DataFrames)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'risk_df': self.risk_df,
                    'streets_gdf': self.streets_gdf,
                    'intersections_gdf': self.intersections_gdf
                }, f)
            
            print(f"Modèle sauvegardé à {self.model_path}")
            return True
        
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle: {e}")
            return False
    
    def load_model(self):
        """
        Charge un modèle entraîné
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            # Charger le modèle
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.risk_df = model_data['risk_df']
                self.streets_gdf = model_data['streets_gdf']
                self.intersections_gdf = model_data['intersections_gdf']
            
            # Tenter de charger le modèle temporel
            try:
                self.time_model.load_model()
            except Exception as e:
                print(f"Avertissement: Impossible de charger le modèle temporel: {e}")
            
            return True
        
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def get_high_risk_zones(self, threshold=0.7, limit=10):
        """
        Retourne les zones à haut risque pour les piétons
        
        Args:
            threshold (float): Seuil de score de risque (0-1)
            limit (int): Nombre maximum de zones à retourner
            
        Returns:
            list: Liste de dictionnaires contenant les informations des zones à risque
        """
        if self.risk_df is None:
            self.load_model()
            if self.risk_df is None:
                return []
        
        # Filtrer les zones à haut risque
        high_risk = self.risk_df[self.risk_df['risk_score'] >= threshold].copy()
        
        # Trier par score de risque décroissant
        high_risk = high_risk.sort_values('risk_score', ascending=False)
        
        # Limiter le nombre de résultats
        high_risk = high_risk.head(limit)
        
        # Créer une liste de dictionnaires avec les informations des zones à risque
        risk_zones = []
        
        for idx, zone in high_risk.iterrows():
            # Obtenir le centroïde de la géométrie
            if hasattr(zone.geometry, 'centroid'):
                centroid = zone.geometry.centroid
                
                zone_info = {
                    'id': int(idx),
                    'type': zone.get('entity_type', 'unknown'),
                    'name': zone.get('full_street_name', 'Zone inconnue') if zone.get('entity_type') == 'street_segment' else f"Intersection {zone.get('intersection_id', idx)}",
                    'risk_score': float(zone.get('risk_score', 0)),
                    'risk_level': str(zone.get('risk_level', 'Inconnu')),
                    'ped_crash_count': int(zone.get('ped_crash_count', 0)),
                    'ped_crash_night_count': int(zone.get('ped_crash_night_count', 0)),
                    'ped_311_count': int(zone.get('ped_311_count', 0)),
                    'sidewalk_issues': int(zone.get('sidewalk_issues', 0)),
                    'crossing_issues': int(zone.get('crossing_issues', 0)),
                    'lighting_issues': int(zone.get('lighting_issues', 0)),
                    'signal_issues': int(zone.get('signal_issues', 0)),
                    'crime_count': int(zone.get('crime_count', 0)),
                    'cluster': int(zone.get('cluster', -1)),
                    'coordinates': {
                        'lat': centroid.y,
                        'lon': centroid.x
                    }
                }
                
                risk_zones.append(zone_info)
        
        return risk_zones
    
    def get_risk_clusters(self, min_cluster_size=2):
        """
        Retourne les informations sur les clusters de risque piéton
        
        Args:
            min_cluster_size (int): Taille minimale d'un cluster pour être inclus
        
        Returns:
            list: Liste de dictionnaires contenant les informations des clusters
        """
        if self.risk_df is None:
            self.load_model()
            if self.risk_df is None:
                return []
        
        # Vérifier si la colonne cluster existe
        if 'cluster' not in self.risk_df.columns:
            return []
        
        # Obtenir les zones avec un cluster valide (pas -1)
        clustered = self.risk_df[self.risk_df['cluster'] >= 0].copy()
        
        # Compter le nombre de zones par cluster
        cluster_sizes = clustered.groupby('cluster').size()
        
        # Filtrer les clusters assez grands
        valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
        
        # Calculer les statistiques par cluster
        cluster_stats = []
        
        for cluster_id in valid_clusters:
            cluster_zones = clustered[clustered['cluster'] == cluster_id]
            
            # Calculer le centroïde du cluster
            cluster_points = [point.centroid for point in cluster_zones.geometry if hasattr(point, 'centroid')]
            
            # S'assurer qu'il y a des points valides
            if not cluster_points:
                continue
                
            cluster_x = np.mean([point.x for point in cluster_points])
            cluster_y = np.mean([point.y for point in cluster_points])
            
            # Calculer des statistiques agrégées
            avg_risk = cluster_zones['risk_score'].mean()
            max_risk = cluster_zones['risk_score'].max()
            total_crashes = cluster_zones['ped_crash_count'].sum()
            total_night_crashes = cluster_zones['ped_crash_night_count'].sum()
            
            # Trouver la zone principale du cluster (celle avec le score de risque le plus élevé)
            main_zone_idx = cluster_zones['risk_score'].idxmax()
            main_zone = cluster_zones.loc[main_zone_idx]
            
            # Déterminer le type de cluster (dominé par des segments de rue ou des intersections)
            n_intersections = (cluster_zones['entity_type'] == 'intersection').sum()
            n_segments = (cluster_zones['entity_type'] == 'street_segment').sum()
            cluster_type = 'intersection' if n_intersections > n_segments else 'street_segment'
            
            # Créer un dictionnaire avec les statistiques du cluster
            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': int(len(cluster_zones)),
                'type': cluster_type,
                'avg_risk_score': float(avg_risk),
                'max_risk_score': float(max_risk),
                'total_pedestrian_crashes': int(total_crashes),
                'total_night_crashes': int(total_night_crashes),
                'main_zone_name': main_zone.get('full_street_name', 'Zone inconnue') if main_zone.get('entity_type') == 'street_segment' else f"Intersection {main_zone.get('intersection_id', main_zone_idx)}",
                'coordinates': {
                    'lat': cluster_y,
                    'lon': cluster_x
                }
            }
            
            cluster_stats.append(cluster_info)
        
        # Trier par score de risque moyen décroissant
        cluster_stats.sort(key=lambda x: x['avg_risk_score'], reverse=True)
        
        return cluster_stats
    
    def get_safety_recommendations(self, zone_id):
        """
        Génère des recommandations pour améliorer la sécurité des piétons dans une zone spécifique
        
        Args:
            zone_id (int): Identifiant de la zone
        
        Returns:
            dict: Dictionnaire contenant les recommandations de sécurité
        """
        if self.risk_df is None:
            self.load_model()
            if self.risk_df is None:
                return {
                    "error": "Aucun modèle chargé"
                }
        
        # Vérifier si la zone existe
        if zone_id not in self.risk_df.index:
            return {
                "error": f"Zone avec ID {zone_id} non trouvée"
            }
        
        # Obtenir les informations sur la zone
        zone = self.risk_df.loc[zone_id]
        
        # Identifier les problèmes spécifiques
        problems = []
        recommendations = []
        
        # Vérifier le type d'entité (segment de rue ou intersection)
        entity_type = zone.get('entity_type', 'unknown')
        
        # Problèmes liés aux accidents
        if zone.get('ped_crash_count', 0) > 0:
            problems.append("Accidents impliquant des piétons")
            
            if entity_type == 'intersection':
                recommendations.append("Améliorer la signalisation aux passages piétons")
                recommendations.append("Installer des feux de signalisation avec phases piétons")
            else:
                recommendations.append("Mettre en place des ralentisseurs de vitesse")
                recommendations.append("Élargir les trottoirs pour améliorer la séparation entre piétons et véhicules")
        
        # Problèmes liés aux accidents de nuit
        if zone.get('ped_crash_night_count', 0) / max(zone.get('ped_crash_count', 1), 1) > 0.5:
            problems.append("Forte proportion d'accidents de nuit")
            recommendations.append("Améliorer l'éclairage public")
            recommendations.append("Installer des bandes réfléchissantes sur les passages piétons")
        
        # Problèmes liés aux trottoirs
        if zone.get('sidewalk_issues', 0) > 0:
            problems.append("Problèmes de trottoirs signalés")
            recommendations.append("Réparer les trottoirs endommagés")
            recommendations.append("Élargir les trottoirs étroits")
        
        # Problèmes liés aux passages piétons
        if zone.get('crossing_issues', 0) > 0:
            problems.append("Problèmes de passages piétons signalés")
            recommendations.append("Installer des passages piétons surélevés")
            recommendations.append("Ajouter des îlots refuges au milieu des grandes traversées")
        
        # Problèmes liés à l'éclairage
        if zone.get('lighting_issues', 0) > 0:
            problems.append("Problèmes d'éclairage signalés")
            recommendations.append("Installer un éclairage LED plus lumineux")
            recommendations.append("Assurer un éclairage spécifique des passages piétons")
        
        # Problèmes liés aux feux de signalisation
        if zone.get('signal_issues', 0) > 0:
            problems.append("Problèmes de signalisation")
            recommendations.append("Réparer ou remplacer les feux de circulation défectueux")
            recommendations.append("Ajouter des comptes à rebours pour piétons aux intersections")
        
        # Problèmes liés à la criminalité
        if zone.get('crime_count', 0) > 0:
            problems.append("Criminalité pouvant affecter la sécurité des piétons")
            recommendations.append("Renforcer la présence policière dans la zone")
            recommendations.append("Installer des caméras de surveillance")
        
        # Recommandations générales basées sur le niveau de risque
        risk_level = zone.get('risk_level', 'Inconnu')
        if risk_level == 'Très élevé':
            recommendations.append("Conduire un audit complet de sécurité piétonne")
            recommendations.append("Mettre en place un programme d'interventions prioritaires")
        elif risk_level == 'Élevé':
            recommendations.append("Envisager des mesures d'apaisement de la circulation")
            
        # Éliminer les doublons dans les recommandations
        recommendations = list(dict.fromkeys(recommendations))
        
        # Obtenir le centroïde de la zone
        centroid = zone.geometry.centroid if hasattr(zone.geometry, 'centroid') else None
        
        return {
            "zone_id": int(zone_id),
            "name": zone.get('full_street_name', 'Zone inconnue') if entity_type == 'street_segment' else f"Intersection {zone.get('intersection_id', zone_id)}",
            "type": entity_type,
            "risk_score": float(zone.get('risk_score', 0)),
            "risk_level": risk_level,
            "problems_identified": problems,
            "recommendations": recommendations,
            "coordinates": {
                "lat": centroid.y if centroid else None,
                "lon": centroid.x if centroid else None
            }
        }
    
    def get_risk_prediction_for_time(self, hour=None, day_of_week=None, month=None, date_time=None):
        """
        Prédit le niveau de risque pour les piétons à un moment spécifique
        
        Args:
            hour (int, optional): Heure de la journée (0-23)
            day_of_week (int/str, optional): Jour de la semaine (0-6 ou nom du jour)
            month (int, optional): Mois (1-12)
            date_time (datetime, optional): Date et heure complètes
            
        Returns:
            dict: Informations de risque pour le moment spécifié
        """
        # Vérifier si le modèle temporel est chargé
        if not hasattr(self, 'time_model') or self.time_model is None:
            self.time_model = TimeBasedPedestrianRiskModel()
            if not self.time_model.load_model():
                print("Aucun modèle temporel disponible. Utilisez train_time_model pour l'entraîner.")
                return {
                    'error': 'Modèle temporel non disponible',
                    'status': 'error'
                }
        
        # Obtenir la prédiction temporelle
        risk_info = self.time_model.predict_risk_at_time(
            hour=hour, day_of_week=day_of_week, month=month, date_time=date_time
        )
        
        # Enrichir avec des informations supplémentaires si disponibles
        if self.risk_df is not None:
            # Rechercher les zones à haut risque avec ce type de risque temporel
            risk_level = risk_info['risk']['level']
            high_risk_zones = self.get_high_risk_zones(threshold=0.7, limit=5)
            
            # Filtrer pour les zones qui correspondent au moment de la journée
            # Par exemple, pour la nuit, prioriser les zones avec des problèmes d'éclairage
            if risk_info['time']['period'] == 'nuit':
                # Trouver les zones avec des problèmes d'éclairage
                night_risk_zones = []
                for zone in high_risk_zones:
                    zone_id = zone.get('id')
                    if zone_id in self.risk_df.index:
                        lighting_issues = self.risk_df.at[zone_id, 'lighting_issues']
                        night_crashes = self.risk_df.at[zone_id, 'ped_crash_night_count']
                        if lighting_issues > 0 or night_crashes > 0:
                            night_risk_zones.append(zone)
                
                # Ajouter ces zones à la réponse
                if night_risk_zones:
                    risk_info['high_risk_zones'] = night_risk_zones[:3]  # Limiter à 3 zones
            
            # Pour les heures de pointe
            elif 7 <= risk_info['time']['hour'] <= 9 or 16 <= risk_info['time']['hour'] <= 18:
                # Trouver les intersections à risque
                rush_hour_zones = []
                for zone in high_risk_zones:
                    if zone.get('type') == 'intersection':
                        rush_hour_zones.append(zone)
                
                # Ajouter ces zones à la réponse
                if rush_hour_zones:
                    risk_info['high_risk_zones'] = rush_hour_zones[:3]
        
        return risk_info
    
    def get_high_risk_times(self, top_n=5):
        """
        Retourne les périodes à plus haut risque pour les piétons
        
        Args:
            top_n (int): Nombre de périodes à retourner
            
        Returns:
            dict: Dictionnaire contenant les périodes à haut risque
        """
        # Vérifier si le modèle temporel est chargé
        if not hasattr(self, 'time_model') or self.time_model is None:
            self.time_model = TimeBasedPedestrianRiskModel()
            if not self.time_model.load_model():
                print("Aucun modèle temporel disponible. Utilisez train_time_model pour l'entraîner.")
                return {
                    'error': 'Modèle temporel non disponible',
                    'status': 'error'
                }
        
        # Obtenir les périodes à haut risque
        return self.time_model.get_high_risk_times(top_n=top_n)
    
    def get_time_risk_heatmap(self):
        """
        Génère les données de heatmap pour le risque par heure et jour de la semaine
        
        Returns:
            dict: Données pour la heatmap
        """
        # Vérifier si le modèle temporel est chargé
        if not hasattr(self, 'time_model') or self.time_model is None:
            self.time_model = TimeBasedPedestrianRiskModel()
            if not self.time_model.load_model():
                print("Aucun modèle temporel disponible. Utilisez train_time_model pour l'entraîner.")
                return {
                    'error': 'Modèle temporel non disponible',
                    'status': 'error'
                }
        
        # Obtenir les données de heatmap
        return self.time_model.get_risk_heatmap_data()