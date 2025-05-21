from flask import Flask, jsonify, request
from datetime import datetime, timedelta

from .risk_model import PedestrianRiskModel
from .geo_utils import create_geojson_from_risk_zones

app = Flask(__name__)
model = PedestrianRiskModel()

# Endpoints de base
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint pour vérifier la santé de l'API"""
    return jsonify({
        'status': 'healthy',
        'message': 'L\'API d\'analyse de la sécurité piétonne est fonctionnelle',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint pour entraîner ou réentraîner le modèle"""
    force_retrain = request.json.get('force_retrain', False) if request.is_json else False
    
    success = model.train(force_retrain=force_retrain)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Modèle entraîné avec succès'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Échec de l\'entraînement du modèle'
        }), 500

@app.route('/risk-zones', methods=['GET'])
def get_risk_zones():
    """
    Endpoint pour obtenir les zones à risque pour les piétons
    
    Query parameters:
        threshold (float): Seuil de score de risque (défaut: 0.7)
        limit (int): Nombre maximum de zones à retourner (défaut: 10)
        format (str): Format de sortie ('json' ou 'geojson', défaut: 'json')
    """
    # Obtenir les paramètres
    threshold = float(request.args.get('threshold', 0.7))
    limit = int(request.args.get('limit', 10))
    output_format = request.args.get('format', 'json').lower()
    
    # Obtenir les zones à risque
    risk_zones = model.get_high_risk_zones(threshold=threshold, limit=limit)
    
    # Formater la sortie
    if output_format == 'geojson':
        geojson = create_geojson_from_risk_zones(risk_zones)
        return jsonify(geojson)
    else:
        return jsonify({
            'count': len(risk_zones),
            'zones': risk_zones
        })

@app.route('/clusters', methods=['GET'])
def get_risk_clusters():
    """
    Endpoint pour obtenir les clusters de zones à risque pour les piétons
    
    Query parameters:
        min_size (int): Taille minimale d'un cluster (défaut: 2)
    """
    min_size = int(request.args.get('min_size', 2))
    
    clusters = model.get_risk_clusters(min_cluster_size=min_size)
    
    return jsonify({
        'count': len(clusters),
        'clusters': clusters
    })

@app.route('/recommendations/<int:zone_id>', methods=['GET'])
def get_safety_recommendations(zone_id):
    """
    Endpoint pour obtenir des recommandations de sécurité pour une zone spécifique
    
    Path parameters:
        zone_id (int): Identifiant de la zone
    """
    recommendations = model.get_safety_recommendations(zone_id)
    
    if 'error' in recommendations:
        return jsonify(recommendations), 404
    
    return jsonify(recommendations)

# Endpoints temporels
@app.route('/time-risk', methods=['GET'])
def get_time_risk():
    """
    Endpoint pour obtenir le risque piéton à un moment spécifique
    
    Query parameters:
        hour (int): Heure (0-23)
        day (int/str): Jour de la semaine (0-6 ou nom du jour)
        month (int): Mois (1-12)
        datetime (str): Date et heure au format ISO (par ex. '2023-05-21T14:30:00')
    """
    # Extraire les paramètres de la requête
    hour = request.args.get('hour')
    day = request.args.get('day')
    month = request.args.get('month')
    datetime_str = request.args.get('datetime')
    
    # Convertir les types si nécessaire
    if hour is not None:
        hour = int(hour)
    if day is not None and day.isdigit():
        day = int(day)
    if month is not None:
        month = int(month)
    
    # Convertir datetime_str en objet datetime si fourni
    datetime_obj = None
    if datetime_str:
        try:
            datetime_obj = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Format de datetime invalide. Utilisez le format ISO (ex: 2023-05-21T14:30:00)'
            }), 400
    
    # Obtenir la prédiction de risque
    risk_info = model.get_risk_prediction_for_time(
        hour=hour,
        day_of_week=day,
        month=month,
        date_time=datetime_obj
    )
    
    # Vérifier s'il y a une erreur
    if isinstance(risk_info, dict) and 'error' in risk_info:
        return jsonify(risk_info), 404
    
    return jsonify(risk_info)

@app.route('/high-risk-times', methods=['GET'])
def get_high_risk_times():
    """
    Endpoint pour obtenir les périodes à plus haut risque pour les piétons
    
    Query parameters:
        top_n (int): Nombre de périodes à retourner par catégorie (défaut: 5)
    """
    top_n = int(request.args.get('top_n', 5))
    
    high_risk_times = model.get_high_risk_times(top_n=top_n)
    
    # Vérifier s'il y a une erreur
    if isinstance(high_risk_times, dict) and 'error' in high_risk_times:
        return jsonify(high_risk_times), 404
    
    return jsonify(high_risk_times)

@app.route('/time-risk-heatmap', methods=['GET'])
def get_time_risk_heatmap():
    """
    Endpoint pour obtenir les données de heatmap du risque temporel
    """
    heatmap_data = model.get_time_risk_heatmap()
    
    # Vérifier s'il y a une erreur
    if isinstance(heatmap_data, dict) and 'error' in heatmap_data:
        return jsonify(heatmap_data), 404
    
    return jsonify(heatmap_data)

@app.route('/current-risk', methods=['GET'])
def get_current_risk():
    """
    Endpoint pour obtenir le risque piéton à l'heure actuelle
    """
    # Obtenir la date et l'heure actuelles
    now = datetime.now()
    
    # Obtenir la prédiction de risque pour le moment actuel
    risk_info = model.get_risk_prediction_for_time(date_time=now)
    
    # Vérifier s'il y a une erreur
    if isinstance(risk_info, dict) and 'error' in risk_info:
        return jsonify({
            'error': risk_info['error'],
            'status': 'error',
            'timestamp': now.isoformat()
        }), 404
    
    # Ajouter l'timestamp actuel à la réponse
    risk_info['timestamp'] = now.isoformat()
    
    return jsonify(risk_info)

@app.route('/risk-forecast', methods=['GET'])
def get_risk_forecast():
    """
    Endpoint pour obtenir les prévisions de risque pour les prochaines heures
    
    Query parameters:
        hours (int): Nombre d'heures à prévoir (défaut: 24)
    """
    hours = int(request.args.get('hours', 24))
    
    # Limiter à un maximum de 48 heures
    if hours > 48:
        hours = 48
    
    # Obtenir l'heure actuelle
    now = datetime.now()
    
    # Générer les prévisions pour chaque heure
    forecast = []
    
    for i in range(hours):
        # Calculer l'heure de prévision
        forecast_time = now + timedelta(hours=i)
        
        # Obtenir la prédiction de risque
        risk_info = model.get_risk_prediction_for_time(date_time=forecast_time)
        
        # Vérifier s'il y a une erreur
        if isinstance(risk_info, dict) and 'error' in risk_info:
            continue
        
        # Simplifier l'objet pour la réponse
        simplified_info = {
            'timestamp': forecast_time.isoformat(),
            'hour': forecast_time.hour,
            'day_of_week': forecast_time.weekday(),
            'day_name': risk_info['time']['day_name'],
            'risk_score': risk_info['risk']['score'],
            'risk_level': risk_info['risk']['level'],
            'period': risk_info['time']['period']
        }
        
        forecast.append(simplified_info)
    
    return jsonify({
        'start_time': now.isoformat(),
        'hours': hours,
        'forecast': forecast
    })

# Endpoints supplémentaires pour des fonctionnalités avancées
@app.route('/statistics', methods=['GET'])
def get_statistics():
    """
    Endpoint pour obtenir des statistiques globales sur la sécurité des piétons
    """
    # Charger le modèle si nécessaire
    if model.risk_df is None:
        model.load_model()
    
    # Vérifier si le modèle est chargé
    if model.risk_df is None:
        return jsonify({
            'status': 'error',
            'message': 'Aucun modèle chargé'
        }), 500
    
    # Nombre total d'accidents de piétons
    total_crashes = model.risk_df['ped_crash_count'].sum()
    
    # Nombre total d'accidents de piétons de nuit
    total_night_crashes = model.risk_df['ped_crash_night_count'].sum()
    
    # Nombre total de signalements liés aux piétons
    total_reports = model.risk_df['ped_311_count'].sum()
    
    # Nombre de zones à risque élevé (score > 0.7)
    high_risk_zones = len(model.risk_df[model.risk_df['risk_score'] >= 0.7])
    
    # Nombre d'intersections dangereuses
    dangerous_intersections = len(model.risk_df[
        (model.risk_df['entity_type'] == 'intersection') & 
        (model.risk_df['risk_score'] >= 0.6)
    ])
    
    # Distribution des niveaux de risque
    risk_level_counts = model.risk_df['risk_level'].value_counts().to_dict()
    
    # Problèmes les plus courants
    sidewalk_issues = model.risk_df['sidewalk_issues'].sum()
    crossing_issues = model.risk_df['crossing_issues'].sum()
    lighting_issues = model.risk_df['lighting_issues'].sum()
    signal_issues = model.risk_df['signal_issues'].sum() if 'signal_issues' in model.risk_df.columns else 0
    
    # Obtenir les périodes à risque
    try:
        high_risk_times = model.get_high_risk_times(top_n=3)
        temporal_data = {
            'high_risk_hours': [time['hour'] for time in high_risk_times['hours']],
            'high_risk_days': [day['day_fr'] for day in high_risk_times['days']],
            'high_risk_months': [month['month_name'] for month in high_risk_times['months']]
        }
    except:
        temporal_data = {
            'high_risk_hours': [],
            'high_risk_days': [],
            'high_risk_months': []
        }
    
    return jsonify({
        'total_pedestrian_crashes': int(total_crashes),
        'total_night_crashes': int(total_night_crashes),
        'total_pedestrian_related_reports': int(total_reports),
        'high_risk_zones_count': int(high_risk_zones),
        'dangerous_intersections_count': int(dangerous_intersections),
        'risk_level_distribution': risk_level_counts,
        'common_issues': {
            'sidewalk_issues': int(sidewalk_issues),
            'crossing_issues': int(crossing_issues),
            'lighting_issues': int(lighting_issues),
            'signal_issues': int(signal_issues)
        },
        'temporal_patterns': temporal_data,
        'updated_at': datetime.now().isoformat()
    })

@app.route('/areas/<string:area_name>', methods=['GET'])
def get_area_risk(area_name):
    """
    Endpoint pour obtenir le risque global pour une zone/quartier spécifique
    
    Path parameters:
        area_name (str): Nom du quartier ou de la zone
    """
    # Charger le modèle si nécessaire
    if model.risk_df is None:
        model.load_model()
    
    # Vérifier si le modèle est chargé
    if model.risk_df is None:
        return jsonify({
            'status': 'error',
            'message': 'Aucun modèle chargé'
        }), 500
    
    # Vérifier si le modèle contient des informations sur les quartiers
    if 'neighborhood' not in model.risk_df.columns:
        return jsonify({
            'status': 'error',
            'message': 'Informations sur les quartiers non disponibles'
        }), 404
    
    # Filtrer pour le quartier spécifié
    area_data = model.risk_df[model.risk_df['neighborhood'].str.contains(area_name, case=False, na=False)]
    
    if len(area_data) == 0:
        return jsonify({
            'status': 'error',
            'message': f'Quartier "{area_name}" non trouvé'
        }), 404
    
    # Calculer des statistiques pour le quartier
    avg_risk = area_data['risk_score'].mean()
    max_risk = area_data['risk_score'].max()
    total_crashes = area_data['ped_crash_count'].sum()
    
    # Obtenir les zones à plus haut risque dans le quartier
    high_risk_in_area = area_data.sort_values('risk_score', ascending=False).head(5)
    
    high_risk_zones = []
    for idx, zone in high_risk_in_area.iterrows():
        # Vérifier si la géométrie a un centroïde
        if hasattr(zone.geometry, 'centroid'):
            centroid = zone.geometry.centroid
            
            zone_info = {
                'id': int(idx),
                'name': zone.get('full_street_name', 'Zone inconnue') if zone.get('entity_type') == 'street_segment' else f"Intersection {zone.get('intersection_id', idx)}",
                'risk_score': float(zone.get('risk_score', 0)),
                'risk_level': str(zone.get('risk_level', 'Inconnu')),
                'coordinates': {
                    'lat': centroid.y,
                    'lon': centroid.x
                }
            }
            
            high_risk_zones.append(zone_info)
    
    return jsonify({
        'area_name': area_name,
        'average_risk_score': float(avg_risk),
        'maximum_risk_score': float(max_risk),
        'total_pedestrian_crashes': int(total_crashes),
        'high_risk_zones': high_risk_zones
    })

@app.route('/compare-times', methods=['GET'])
def compare_times():
    """
    Endpoint pour comparer le risque entre différents moments
    
    Query parameters:
        time1 (str): Premier moment au format ISO (ex: '2023-05-21T14:30:00')
        time2 (str): Deuxième moment au format ISO (ex: '2023-05-21T20:30:00')
    """
    # Extraire les paramètres
    time1_str = request.args.get('time1')
    time2_str = request.args.get('time2')
    
    # Vérifier que les deux temps sont fournis
    if not time1_str or not time2_str:
        return jsonify({
            'status': 'error',
            'message': 'Les deux paramètres time1 et time2 sont requis'
        }), 400
    
    # Convertir en objets datetime
    try:
        time1 = datetime.fromisoformat(time1_str.replace('Z', '+00:00'))
        time2 = datetime.fromisoformat(time2_str.replace('Z', '+00:00'))
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Format de datetime invalide. Utilisez le format ISO (ex: 2023-05-21T14:30:00)'
        }), 400
    
    # Obtenir les prédictions pour les deux moments
    risk1 = model.get_risk_prediction_for_time(date_time=time1)
    risk2 = model.get_risk_prediction_for_time(date_time=time2)
    
    # Vérifier s'il y a des erreurs
    if isinstance(risk1, dict) and 'error' in risk1:
        return jsonify(risk1), 404
    if isinstance(risk2, dict) and 'error' in risk2:
        return jsonify(risk2), 404
    
    # Calculer la différence de risque
    risk_diff = risk2['risk']['score'] - risk1['risk']['score']
    
    return jsonify({
        'time1': {
            'datetime': time1.isoformat(),
            'hour': time1.hour,
            'day': time1.strftime('%A'),
            'risk_score': risk1['risk']['score'],
            'risk_level': risk1['risk']['level'],
            'period': risk1['time']['period']
        },
        'time2': {
            'datetime': time2.isoformat(),
            'hour': time2.hour,
            'day': time2.strftime('%A'),
            'risk_score': risk2['risk']['score'],
            'risk_level': risk2['risk']['level'],
            'period': risk2['time']['period']
        },
        'comparison': {
            'risk_difference': risk_diff,
            'percentage_change': (risk_diff / risk1['risk']['score']) * 100 if risk1['risk']['score'] > 0 else 0,
            'safer_time': 'time1' if risk1['risk']['score'] < risk2['risk']['score'] else 'time2',
            'recommendation': f"Pour une sécurité optimale, privilégiez {time1.strftime('%A à %H:%M')}" if risk1['risk']['score'] < risk2['risk']['score'] else f"Pour une sécurité optimale, privilégiez {time2.strftime('%A à %H:%M')}"
        }
    })

def start_api(host='0.0.0.0', port=8000, debug=False):
    """
    Démarre l'API Flask
    
    Args:
        host (str): Hôte sur lequel démarrer l'API
        port (int): Port sur lequel démarrer l'API
        debug (bool): Mode debug
    """
    # Charger le modèle
    try:
        model.load_model()
        print(f"Modèle chargé avec succès")
    except Exception as e:
        print(f"Avertissement: Impossible de charger le modèle: {e}")
        print("L'API fonctionnera, mais certains endpoints nécessiteront un entraînement préalable")
    
    # Démarrer l'API
    print(f"Démarrage de l'API sur {host}:{port}...")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Si exécuté directement, démarrer l'API en mode debug
    start_api(debug=True)