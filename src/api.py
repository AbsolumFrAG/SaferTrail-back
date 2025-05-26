from pathlib import Path
from flask import Flask, jsonify, request, send_file
from datetime import datetime, timedelta
from flask_swagger_ui import get_swaggerui_blueprint

from .risk_model import PedestrianRiskModel
from .geo_utils import create_geojson_from_risk_zones, create_geojson_from_street_segments
from .alt_routing import ALTRouter

app = Flask(__name__)
model = PedestrianRiskModel()

SWAGGER_URL = '/docs'
API_URL = '/swagger.yaml'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "SaferTrail API"}
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Initialiser le routeur avec le modèle de risque
router = ALTRouter(model)

def initialize_router_on_startup():
    """Initialise le routeur ALT avec les données du modèle"""
    try:
        if model.risk_df is not None and model.streets_gdf is not None:
            router.build_graph(model.risk_df, model.streets_gdf)
            print("Routeur ALT initialisé avec succès")
        else:
            print("Les données du modèle ne sont pas disponibles, le routeur sera initialisé à la demande")
    except Exception as e:
        print(f"Erreur lors de l'initialisation du routeur: {e}")
        print("Le routeur sera initialisé lors de la première requête")

@app.route("/swagger.yaml")
def swagger_spec():
    try:
        # Chemin vers la racine du projet (un niveau au-dessus du dossier src)
        project_root = Path(__file__).parent.parent
        swagger_file = project_root / 'swagger.yaml'
        
        if not swagger_file.exists():
            return jsonify({
                'error': 'Fichier swagger.yaml non trouvé',
                'expected_path': str(swagger_file)
            }), 404
        
        return send_file(str(swagger_file), mimetype='text/yaml')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoints de base
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint pour vérifier la santé de l'API"""
    return jsonify({
        'status': 'healthy',
        'message': 'L\'API d\'analyse de la sécurité piétonne est fonctionnelle',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'risk_analysis': True,
            'temporal_prediction': True,
            'safe_routing': router.graph is not None
        }
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint pour entraîner ou réentraîner le modèle"""
    force_retrain = request.json.get('force_retrain', False) if request.is_json else False
    
    success = model.train(force_retrain=force_retrain)
    
    if success:
        # Réinitialiser le routeur avec les nouvelles données
        try:
            router.build_graph(model.risk_df, model.streets_gdf)
            routing_status = "Routeur mis à jour"
        except Exception as e:
            routing_status = f"Erreur lors de la mise à jour du routeur: {e}"
        
        return jsonify({
            'status': 'success',
            'message': 'Modèle entraîné avec succès',
            'routing_status': routing_status
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

# Endpoints de routage ALT
@app.route('/safe-route', methods=['GET'])
def get_safe_route():
    """
    Endpoint pour obtenir un itinéraire sécurisé pour les piétons
    
    Query parameters:
        start_lat (float): Latitude du point de départ
        start_lon (float): Longitude du point de départ
        end_lat (float): Latitude du point d'arrivée
        end_lon (float): Longitude du point d'arrivée
        time (str, optional): Heure à laquelle effectuer le routage (format ISO, ex: '2023-05-21T14:30:00')
        risk_weight (float, optional): Importance du facteur de risque vs distance (0-1)
        format (str, optional): Format de sortie ('json' ou 'geojson', défaut: 'json')
    """
    # Extraire les paramètres
    try:
        start_lat = float(request.args.get('start_lat'))
        start_lon = float(request.args.get('start_lon'))
        end_lat = float(request.args.get('end_lat'))
        end_lon = float(request.args.get('end_lon'))
    except (TypeError, ValueError):
        return jsonify({
            'status': 'error',
            'message': 'Coordonnées invalides. Veuillez fournir des valeurs numériques pour start_lat, start_lon, end_lat, end_lon.'
        }), 400
    
    # Paramètres optionnels
    time_str = request.args.get('time')
    risk_weight = request.args.get('risk_weight')
    output_format = request.args.get('format', 'json').lower()
    
    # Convertir l'heure si spécifiée
    time = None
    if time_str:
        try:
            time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Format d\'heure invalide. Utilisez le format ISO (ex: 2023-05-21T14:30:00)'
            }), 400
    else:
        # Utiliser l'heure actuelle par défaut
        time = datetime.now()
    
    # Définir le poids du risque si spécifié
    if risk_weight is not None:
        try:
            router.set_risk_weight(float(risk_weight))
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Le poids du risque doit être entre 0 et 1'
            }), 400
    
    # Initialiser le graphe si ce n'est pas déjà fait
    if router.graph is None:
        try:
            if model.risk_df is None:
                model.load_model()
            
            router.build_graph(model.risk_df, model.streets_gdf)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Erreur lors de la construction du graphe: {str(e)}'
            }), 500
    
    # Points de départ et d'arrivée
    start_point = (start_lon, start_lat)
    end_point = (end_lon, end_lat)
    
    try:
        # Calculer l'itinéraire
        if output_format == 'geojson':
            result = router.get_route_as_geojson(start_point, end_point, time)
            return jsonify(result)
        else:
            result = router.get_safe_route(start_point, end_point, time)
            return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erreur lors du calcul de l\'itinéraire: {str(e)}'
        }), 500

@app.route('/compare-routes', methods=['GET'])
def compare_routes():
    """
    Endpoint pour comparer un itinéraire sécurisé vs le plus court
    
    Query parameters:
        start_lat (float): Latitude du point de départ
        start_lon (float): Longitude du point de départ
        end_lat (float): Latitude du point d'arrivée
        end_lon (float): Longitude du point d'arrivée
        time (str, optional): Heure à laquelle effectuer le routage (format ISO, ex: '2023-05-21T14:30:00')
    """
    # Extraire les paramètres
    try:
        start_lat = float(request.args.get('start_lat'))
        start_lon = float(request.args.get('start_lon'))
        end_lat = float(request.args.get('end_lat'))
        end_lon = float(request.args.get('end_lon'))
    except (TypeError, ValueError):
        return jsonify({
            'status': 'error',
            'message': 'Coordonnées invalides. Veuillez fournir des valeurs numériques pour start_lat, start_lon, end_lat, end_lon.'
        }), 400
    
    # Paramètres optionnels
    time_str = request.args.get('time')
    
    # Convertir l'heure si spécifiée
    time = None
    if time_str:
        try:
            time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Format d\'heure invalide. Utilisez le format ISO (ex: 2023-05-21T14:30:00)'
            }), 400
    else:
        # Utiliser l'heure actuelle par défaut
        time = datetime.now()
    
    # Initialiser le graphe si ce n'est pas déjà fait
    if router.graph is None:
        try:
            if model.risk_df is None:
                model.load_model()
            
            router.build_graph(model.risk_df, model.streets_gdf)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Erreur lors de la construction du graphe: {str(e)}'
            }), 500
    
    # Points de départ et d'arrivée
    start_point = (start_lon, start_lat)
    end_point = (end_lon, end_lat)
    
    try:
        # Sauvegarder le paramètre de risque actuel
        original_risk_weight = router.risk_weight
        
        # Calculer l'itinéraire sécurisé (risque important)
        router.set_risk_weight(0.7)  # Favorise la sécurité
        safe_route = router.get_safe_route(start_point, end_point, time)
        
        # Calculer l'itinéraire le plus court (distance uniquement)
        router.set_risk_weight(0.0)  # Ignore le risque
        shortest_route = router.get_safe_route(start_point, end_point, time)
        
        # Restaurer le paramètre original
        router.set_risk_weight(original_risk_weight)
        
        # Comparer les deux itinéraires
        if safe_route["status"] == "error" or shortest_route["status"] == "error":
            return jsonify({
                'status': 'error',
                'message': 'Impossible de calculer les deux itinéraires pour la comparaison'
            }), 500
            
        # Calculer les différences
        safe_distance = safe_route["route"]["distance"]
        shortest_distance = shortest_route["route"]["distance"]
        safe_risk = safe_route["route"]["risk_score"]
        shortest_risk = shortest_route["route"]["risk_score"]
        
        distance_diff = safe_distance - shortest_distance
        distance_percent = (distance_diff / shortest_distance) * 100 if shortest_distance > 0 else 0
        risk_diff = shortest_risk - safe_risk
        risk_percent = (risk_diff / shortest_risk) * 100 if shortest_risk > 0 else 0
        
        # Créer la réponse
        comparison = {
            'status': 'success',
            'safe_route': safe_route["route"],
            'shortest_route': shortest_route["route"],
            'comparison': {
                'distance_difference': distance_diff,
                'distance_percentage': distance_percent,
                'risk_difference': risk_diff,
                'risk_percentage': risk_percent,
                'safer_route_worth_it': risk_percent >= distance_percent
            },
            'recommendation': "L'itinéraire le plus sûr représente un bon compromis entre sécurité et distance." 
                              if risk_percent >= distance_percent else
                              "L'itinéraire le plus court peut être préférable, le détour pour plus de sécurité étant important."
        }
        
        return jsonify(comparison)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erreur lors de la comparaison des itinéraires: {str(e)}'
        }), 500

@app.route('/routing-settings', methods=['GET', 'POST'])
def routing_settings():
    """
    Endpoint pour obtenir ou modifier les paramètres de routage
    """
    if request.method == 'GET':
        # Retourner les paramètres actuels
        return jsonify({
            'risk_weight': router.risk_weight,
            'max_detour_factor': router.max_detour_factor,
            'graph_initialized': router.graph is not None,
            'num_landmarks': len(router.landmarks) if router.landmarks else 0
        })
    else:  # POST
        # Mettre à jour les paramètres
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Le corps de la requête doit être en JSON'
            }), 400
            
        data = request.json
        
        # Mettre à jour le poids du risque
        if 'risk_weight' in data:
            try:
                router.set_risk_weight(float(data['risk_weight']))
            except ValueError as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 400
        
        # Mettre à jour le facteur de détour maximum
        if 'max_detour_factor' in data:
            try:
                router.set_max_detour_factor(float(data['max_detour_factor']))
            except ValueError as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 400
        
        return jsonify({
            'status': 'success',
            'message': 'Paramètres mis à jour',
            'risk_weight': router.risk_weight,
            'max_detour_factor': router.max_detour_factor
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
        'routing_available': router.graph is not None,
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

@app.route('/street-segments', methods=['GET'])
def get_street_segments():
    """
    Endpoint pour obtenir tous les segments de rue avec leurs scores de risque
    
    Query parameters:
        color_by_risk (bool): Si True, ajoute une couleur basée sur le niveau de risque (défaut: True)
        min_risk (float): Score de risque minimum pour inclure un segment (défaut: 0)
        limit (int): Nombre maximum de segments à inclure (défaut: 1000)
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
    
    # Obtenir les paramètres
    color_by_risk = request.args.get('color_by_risk', 'true').lower() == 'true'
    min_risk = float(request.args.get('min_risk', 0))
    limit = int(request.args.get('limit', 1000))
    
    # Créer le GeoJSON
    geojson = create_geojson_from_street_segments(
        model.risk_df, 
        color_by_risk=color_by_risk,
        min_risk=min_risk,
        limit=limit
    )
    
    return jsonify(geojson)

# Endpoint de gestion du routeur
@app.route('/router-status', methods=['GET'])
def get_router_status():
    """
    Endpoint pour obtenir le statut du routeur ALT
    """
    return jsonify({
        'router_initialized': router.graph is not None,
        'graph_nodes': len(router.graph.nodes) if router.graph else 0,
        'graph_edges': len(router.graph.edges) if router.graph else 0,
        'landmarks_count': len(router.landmarks),
        'risk_weight': router.risk_weight,
        'max_detour_factor': router.max_detour_factor
    })

@app.route('/initialize-router', methods=['POST'])
def initialize_router():
    """
    Endpoint pour forcer l'initialisation du routeur
    """
    try:
        if model.risk_df is None:
            model.load_model()
        
        router.build_graph(model.risk_df, model.streets_gdf)
        
        return jsonify({
            'status': 'success',
            'message': 'Routeur initialisé avec succès',
            'graph_nodes': len(router.graph.nodes),
            'graph_edges': len(router.graph.edges),
            'landmarks_count': len(router.landmarks)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erreur lors de l\'initialisation du routeur: {str(e)}'
        }), 500

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
        
        # Initialiser le routeur
        initialize_router_on_startup()
    except Exception as e:
        print(f"Avertissement: Impossible de charger le modèle: {e}")
        print("L'API fonctionnera, mais certains endpoints nécessiteront un entraînement préalable")
    
    # Démarrer l'API
    print(f"Démarrage de l'API sur {host}:{port}...")
    print("\nEndpoints disponibles:")
    print("- Analyse de risque: /street-segments, /risk-zones, /clusters, /recommendations/<id>")
    print("- Analyse temporelle: /time-risk, /current-risk, /risk-forecast")
    print("- Routage sécurisé: /safe-route, /compare-routes")
    print("- Statistiques: /statistics, /areas/<name>")
    print("- Configuration: /routing-settings, /router-status")
    print("- Santé: /health")
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Si exécuté directement, démarrer l'API en mode debug
    start_api(debug=True)