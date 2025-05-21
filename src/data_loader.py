import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime, timedelta

def load_crash_data(file_path="data/vision-zero-crash-records.csv", pedestrian_only=True):
    """
    Charge les données d'accidents de la route, avec option de filtrer pour les piétons
    
    Args:
        file_path (str): Chemin vers le fichier CSV des données d'accidents
        pedestrian_only (bool): Si True, filtre uniquement les accidents impliquant des piétons
    
    Returns:
        DataFrame: Données d'accidents chargées et filtrées
    """
    try:
        df_crashes = pd.read_csv(file_path, sep=",", encoding="utf-8")
        
        # Convertir les dates en datetime avec gestion des timezones
        df_crashes['dispatch_ts'] = pd.to_datetime(df_crashes['dispatch_ts'], errors='coerce')
        
        # Vérifier si les dates ont un fuseau horaire et normaliser
        if hasattr(df_crashes['dispatch_ts'].dt, 'tz') and df_crashes['dispatch_ts'].dt.tz is not None:
            # Si les dates ont un fuseau horaire, nous utilisons une date de comparaison avec le même fuseau
            timezone_info = df_crashes['dispatch_ts'].dt.tz
            print(f"Les dates ont un fuseau horaire: {timezone_info}")
        else:
            # Si les dates n'ont pas de fuseau horaire, nous les laissons naïves
            print("Les dates n'ont pas de fuseau horaire")
        
        # Créer des colonnes pour l'analyse temporelle
        df_crashes['year'] = df_crashes['dispatch_ts'].dt.year
        df_crashes['month'] = df_crashes['dispatch_ts'].dt.month
        df_crashes['hour'] = df_crashes['dispatch_ts'].dt.hour
        df_crashes['day_of_week'] = df_crashes['dispatch_ts'].dt.day_name()
        
        # Filtrer pour les accidents récents (3 dernières années)
        three_years_ago = datetime.now() - timedelta(days=3*365)
        
        # Si les dates ont un fuseau horaire, adapter la date de comparaison
        if hasattr(df_crashes['dispatch_ts'].dt, 'tz') and df_crashes['dispatch_ts'].dt.tz is not None:
            import pytz
            # Convertir la date de comparaison pour qu'elle ait le même fuseau horaire
            timezone_name = str(df_crashes['dispatch_ts'].dt.tz)
            try:
                timezone = pytz.timezone(timezone_name)
                three_years_ago = timezone.localize(three_years_ago)
            except:
                # Si nous ne pouvons pas localiser, nous convertissons les dates en naïves
                print(f"Conversion des dates UTC en dates naïves pour comparaison")
                df_crashes['dispatch_ts'] = df_crashes['dispatch_ts'].dt.tz_localize(None)
        
        # Filtrer par date
        df_crashes = df_crashes[df_crashes['dispatch_ts'] >= three_years_ago]
        
        # Filtrer uniquement pour les accidents impliquant des piétons si demandé
        if pedestrian_only:
            df_crashes = df_crashes[df_crashes['mode_type'] == 'ped'].copy()
        
        # Garder seulement les entrées avec des coordonnées valides
        df_crashes = df_crashes.dropna(subset=['lat', 'long'])
        
        print(f"Données d'accidents chargées: {len(df_crashes)} entrées")
        return df_crashes
    
    except Exception as e:
        print(f"Erreur lors du chargement des données d'accidents: {e}")
        return pd.DataFrame()

def load_311_data(file_path="data/311.csv", pedestrian_only=True):
    """
    Charge les données 311, avec option de filtrer pour les problèmes liés aux piétons
    
    Args:
        file_path (str): Chemin vers le fichier CSV des données 311
        pedestrian_only (bool): Si True, filtre uniquement les signalements liés aux piétons
    
    Returns:
        DataFrame: Données 311 chargées et filtrées
    """
    try:
        # Charger les données en spécifiant certains types pour éviter les problèmes de parsing
        df_311 = pd.read_csv(file_path, sep=",", encoding="utf-8", low_memory=False)
        
        # Convertir les colonnes de date en datetime
        for col in ['open_dt', 'closed_dt', 'sla_target_dt']:
            if col in df_311.columns:
                df_311[col] = pd.to_datetime(df_311[col], errors='coerce')
        
        # Vérifier si les dates ont un fuseau horaire et normaliser
        if 'open_dt' in df_311.columns and hasattr(df_311['open_dt'].dt, 'tz') and df_311['open_dt'].dt.tz is not None:
            # Si les dates ont un fuseau horaire, nous utilisons une date de comparaison avec le même fuseau
            timezone_info = df_311['open_dt'].dt.tz
            print(f"Les dates 311 ont un fuseau horaire: {timezone_info}")
        else:
            # Si les dates n'ont pas de fuseau horaire, nous les laissons naïves
            print("Les dates 311 n'ont pas de fuseau horaire")
        
        # Créer des colonnes pour l'analyse temporelle
        df_311['year'] = df_311['open_dt'].dt.year
        df_311['month'] = df_311['open_dt'].dt.month
        df_311['hour'] = df_311['open_dt'].dt.hour
        df_311['day_of_week'] = df_311['open_dt'].dt.day_name()
        
        # Filtrer pour les signalements récents (3 dernières années)
        three_years_ago = datetime.now() - timedelta(days=3*365)
        
        # Si les dates ont un fuseau horaire, adapter la date de comparaison
        if 'open_dt' in df_311.columns and hasattr(df_311['open_dt'].dt, 'tz') and df_311['open_dt'].dt.tz is not None:
            import pytz
            # Convertir la date de comparaison pour qu'elle ait le même fuseau horaire
            timezone_name = str(df_311['open_dt'].dt.tz)
            try:
                timezone = pytz.timezone(timezone_name)
                three_years_ago = timezone.localize(three_years_ago)
            except:
                # Si nous ne pouvons pas localiser, nous convertissons les dates en naïves
                print(f"Conversion des dates 311 en dates naïves pour comparaison")
                df_311['open_dt'] = df_311['open_dt'].dt.tz_localize(None)
        
        # Filtrer par date
        df_311 = df_311[df_311['open_dt'] >= three_years_ago]
        
        # Filtrer pour les signalements liés aux piétons si demandé
        if pedestrian_only:
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
            
            df_311 = df_311[df_311['ped_related']].copy()
        
        # Garder seulement les entrées avec des coordonnées valides
        df_311 = df_311.dropna(subset=['latitude', 'longitude'])
        
        print(f"Données 311 chargées: {len(df_311)} entrées")
        return df_311
    
    except Exception as e:
        print(f"Erreur lors du chargement des données 311: {e}")
        return pd.DataFrame()

def load_street_segments(file_path="data/boston-street-segments.geojson"):
    """
    Charge les données des segments de rue
    
    Args:
        file_path (str): Chemin vers le fichier GeoJSON des segments de rue
    
    Returns:
        GeoDataFrame: Données des segments de rue chargées
    """
    try:
        gdf_streets = gpd.read_file(file_path)
        
        # Assurer que la projection est en WGS84 (EPSG:4326)
        if gdf_streets.crs != "EPSG:4326":
            gdf_streets = gdf_streets.to_crs("EPSG:4326")
        
        # Créer une colonne avec le nom complet de la rue
        gdf_streets['full_street_name'] = gdf_streets.apply(
            lambda x: ' '.join(filter(None, [
                x.get('PRE_DIR', ''), 
                x.get('ST_NAME', ''), 
                x.get('ST_TYPE', ''), 
                x.get('SUF_DIR', '')
            ])),
            axis=1
        )
        
        print(f"Données des segments de rue chargées: {len(gdf_streets)} segments")
        return gdf_streets
    
    except Exception as e:
        print(f"Erreur lors du chargement des segments de rue: {e}")
        return gpd.GeoDataFrame()

def load_crime_data(file_path="data/crimes-incident-report.csv", pedestrian_related=True):
    """
    Charge les données des incidents criminels
    
    Args:
        file_path (str): Chemin vers le fichier CSV des données criminelles
        pedestrian_related (bool): Si True, filtre pour les crimes pouvant affecter la sécurité des piétons
    
    Returns:
        DataFrame: Données criminelles chargées et filtrées
    """
    try:
        df_crimes = pd.read_csv(file_path, sep=",", encoding="utf-8")
        
        # Convertir les dates en datetime
        df_crimes['OCCURRED_ON_DATE'] = pd.to_datetime(df_crimes['OCCURRED_ON_DATE'], errors='coerce')
        
        # Vérifier si les dates ont un fuseau horaire et normaliser
        if hasattr(df_crimes['OCCURRED_ON_DATE'].dt, 'tz') and df_crimes['OCCURRED_ON_DATE'].dt.tz is not None:
            # Si les dates ont un fuseau horaire, nous utilisons une date de comparaison avec le même fuseau
            timezone_info = df_crimes['OCCURRED_ON_DATE'].dt.tz
            print(f"Les dates des crimes ont un fuseau horaire: {timezone_info}")
        else:
            # Si les dates n'ont pas de fuseau horaire, nous les laissons naïves
            print("Les dates des crimes n'ont pas de fuseau horaire")
        
        # Créer des colonnes pour l'analyse temporelle
        df_crimes['year'] = df_crimes['OCCURRED_ON_DATE'].dt.year
        df_crimes['month'] = df_crimes['OCCURRED_ON_DATE'].dt.month
        df_crimes['hour'] = df_crimes['OCCURRED_ON_DATE'].dt.hour
        df_crimes['day_of_week'] = df_crimes['OCCURRED_ON_DATE'].dt.day_name()
        
        # Filtrer pour les crimes récents (3 dernières années)
        three_years_ago = datetime.now() - timedelta(days=3*365)
        
        # Si les dates ont un fuseau horaire, adapter la date de comparaison
        if hasattr(df_crimes['OCCURRED_ON_DATE'].dt, 'tz') and df_crimes['OCCURRED_ON_DATE'].dt.tz is not None:
            import pytz
            # Convertir la date de comparaison pour qu'elle ait le même fuseau horaire
            timezone_name = str(df_crimes['OCCURRED_ON_DATE'].dt.tz)
            try:
                timezone = pytz.timezone(timezone_name)
                three_years_ago = timezone.localize(three_years_ago)
            except:
                # Si nous ne pouvons pas localiser, nous convertissons les dates en naïves
                print(f"Conversion des dates des crimes en dates naïves pour comparaison")
                df_crimes['OCCURRED_ON_DATE'] = df_crimes['OCCURRED_ON_DATE'].dt.tz_localize(None)
        
        # Filtrer par date
        df_crimes = df_crimes[df_crimes['OCCURRED_ON_DATE'] >= three_years_ago]
        
        # Filtrer pour les crimes pouvant affecter la sécurité des piétons
        if pedestrian_related:
            ped_safety_keywords = [
                'ASSAULT', 'ROBBERY', 'LARCENY', 'HARASSMENT', 'THREATS', 
                'DISORDERLY', 'VANDALISM', 'PUBLIC DRINKING'
            ]
            
            df_crimes['ped_safety_related'] = df_crimes['OFFENSE_DESCRIPTION'].str.contains(
                '|'.join(ped_safety_keywords), case=False, na=False
            )
            
            df_crimes = df_crimes[df_crimes['ped_safety_related']].copy()
        
        # Garder seulement les entrées avec des coordonnées valides
        df_crimes = df_crimes.dropna(subset=['Lat', 'Long'])
        
        print(f"Données criminelles chargées: {len(df_crimes)} entrées")
        return df_crimes
    
    except Exception as e:
        print(f"Erreur lors du chargement des données criminelles: {e}")
        return pd.DataFrame()

def load_all_data(pedestrian_focus=True):
    """
    Charge toutes les données nécessaires pour le modèle, avec option de focus sur les piétons
    
    Args:
        pedestrian_focus (bool): Si True, filtre les données pour se concentrer sur la sécurité des piétons
    
    Returns:
        tuple: Tuple contenant tous les DataFrames chargés
    """
    data_dir = Path(__file__).parent.parent / 'data'
    
    df_crashes = load_crash_data(data_dir / 'vision-zero-crash-records.csv', pedestrian_only=pedestrian_focus)
    df_311 = load_311_data(data_dir / '311.csv', pedestrian_only=pedestrian_focus)
    gdf_streets = load_street_segments(data_dir / 'boston-street-segments.geojson')
    df_crimes = load_crime_data(data_dir / 'crimes-incident-report.csv', pedestrian_related=pedestrian_focus)
    
    return df_crashes, df_311, gdf_streets, df_crimes