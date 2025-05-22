import networkx as nx
from shapely.geometry import Point, LineString
from datetime import datetime
import heapq
import random
import time


def convert_degrees_to_miles(distance_degrees):
    """
    Convertit une distance en degrés géographiques en miles
    
    Args:
        distance_degrees (float): Distance en degrés
        
    Returns:
        float: Distance en miles
    """
    # 1 degré de latitude/longitude ≈ 68.97 miles (moyenne)
    miles = distance_degrees * 68.97
    return miles


def convert_degrees_to_meters(distance_degrees):
    """
    Convertit une distance en degrés géographiques en mètres
    
    Args:
        distance_degrees (float): Distance en degrés
        
    Returns:
        float: Distance en mètres
    """
    # 1 degré ≈ 111 km = 111,000 mètres
    meters = distance_degrees * 111000
    return meters


def format_walking_time(seconds):
    """
    Formate le temps de marche en heures et minutes
    
    Args:
        seconds (float): Temps en secondes
        
    Returns:
        dict: Dictionnaire avec le temps formaté
    """
    total_minutes = int(seconds / 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    if hours > 0:
        if minutes > 0:
            formatted = f"{hours}h {minutes}min"
        else:
            formatted = f"{hours}h"
    else:
        formatted = f"{minutes}min"
    
    return {
        'total_seconds': int(seconds),
        'total_minutes': total_minutes,
        'hours': hours,
        'minutes': minutes,
        'formatted': formatted
    }


def classify_road_for_pedestrians(segment):
    """
    Classifie une route selon sa praticabilité pour les piétons
    
    Args:
        segment: Données du segment de route
        
    Returns:
        dict: Classification détaillée
    """
    # Récupérer les attributs du segment
    street_name = str(segment.get('full_street_name', '')).upper()
    st_type = str(segment.get('ST_TYPE', '')).upper()
    cfcc = str(segment.get('CFCC', ''))
    speed_limit = segment.get('SPEEDLIMIT', 0)
    highway_num = str(segment.get('HWY_NUM', ''))
    
    # Classification par défaut
    classification = {
        'walkable': True,
        'category': 'normal',
        'restrictions': [],
        'warnings': [],
        'requires_transport': False,
        'transport_type': None,
        'pedestrian_safety': 'safe'
    }
    
    # ==========================================
    # ROUTES INTERDITES AUX PIÉTONS
    # ==========================================
    
    # Autoroutes et voies rapides
    highway_keywords = [
        'INTERSTATE', 'I-', 'I ', 'HIGHWAY', 'HWY', 'EXPRESSWAY', 'FREEWAY',
        'TURNPIKE', 'PARKWAY', 'THRUWAY'
    ]
    
    # Types de routes interdites (CFCC codes)
    prohibited_cfcc = [
        'A11',  # Primary highway with limited access (Interstate)
        'A12',  # Primary highway with limited access (Interstate, business route)
        'A13',  # Primary highway with limited access (Interstate, business spur)
        'A14',  # Primary highway with limited access (US highway)
        'A15',  # Primary highway with limited access (State highway)
        'A21',  # Primary highway without limited access (US highway)
    ]
    
    # Vérifier les mots-clés d'autoroute
    for keyword in highway_keywords:
        if keyword in street_name:
            classification.update({
                'walkable': False,
                'category': 'highway',
                'restrictions': ['no_pedestrians'],
                'pedestrian_safety': 'prohibited'
            })
            return classification
    
    # Vérifier les codes CFCC
    if cfcc in prohibited_cfcc:
        classification.update({
            'walkable': False,
            'category': 'highway',
            'restrictions': ['no_pedestrians', 'limited_access'],
            'pedestrian_safety': 'prohibited'
        })
        return classification
    
    # Vérifier la vitesse limite (> 55 mph généralement interdit aux piétons)
    if speed_limit and speed_limit > 55:
        classification.update({
            'walkable': False,
            'category': 'high_speed',
            'restrictions': ['no_pedestrians', 'high_speed'],
            'pedestrian_safety': 'prohibited'
        })
        return classification
    
    # ==========================================
    # TRANSPORTS SPÉCIAUX REQUIS
    # ==========================================
    
    ferry_keywords = ['FERRY', 'BOAT', 'WATER TAXI', 'DOCK']
    bridge_keywords = ['BRIDGE', 'PONT', 'OVERPASS', 'TUNNEL', 'UNDERPASS']
    transit_keywords = ['SUBWAY', 'METRO', 'BUS RAPID', 'LIGHT RAIL', 'TROLLEY']
    
    # Ferry ou transport maritime
    for keyword in ferry_keywords:
        if keyword in street_name:
            classification.update({
                'walkable': True,
                'category': 'ferry',
                'requires_transport': True,
                'transport_type': 'ferry',
                'warnings': ['requires_ferry_ticket', 'schedule_dependent'],
                'pedestrian_safety': 'safe_with_transport'
            })
            return classification
    
    # Ponts majeurs ou tunnels (peuvent nécessiter transport public)
    major_bridges = [
        'TOBIN', 'ZAKIM', 'LONGFELLOW', 'HARVARD', 'MASS AVE BRIDGE',
        'BU BRIDGE', 'WESTERN AVE BRIDGE', 'TUNNEL'
    ]
    
    for bridge in major_bridges:
        if bridge in street_name:
            # Certains ponts ont des restrictions piétonnes
            if any(x in street_name for x in ['TOBIN', 'TUNNEL']):
                classification.update({
                    'walkable': False,
                    'category': 'restricted_bridge',
                    'restrictions': ['no_pedestrians', 'vehicle_only'],
                    'requires_transport': True,
                    'transport_type': 'vehicle_required',
                    'pedestrian_safety': 'prohibited'
                })
            else:
                classification.update({
                    'walkable': True,
                    'category': 'bridge',
                    'warnings': ['wind_exposure', 'weather_dependent'],
                    'pedestrian_safety': 'caution'
                })
            return classification
    
    # Transport en commun dédié
    for keyword in transit_keywords:
        if keyword in street_name:
            classification.update({
                'walkable': True,
                'category': 'transit',
                'requires_transport': True,
                'transport_type': 'public_transit',
                'warnings': ['requires_ticket', 'schedule_dependent'],
                'pedestrian_safety': 'safe_with_transport'
            })
            return classification
    
    # ==========================================
    # ROUTES AVEC AVERTISSEMENTS
    # ==========================================
    
    # Routes avec vitesse élevée mais praticables
    if speed_limit and 35 < speed_limit <= 55:
        classification.update({
            'category': 'arterial',
            'warnings': ['high_traffic', 'limited_sidewalks'],
            'pedestrian_safety': 'caution'
        })
    
    # Types de routes spéciales
    special_types = {
        'ALY': {'category': 'alley', 'warnings': ['narrow', 'limited_lighting']},
        'CT': {'category': 'court', 'warnings': ['dead_end']},
        'PLZ': {'category': 'plaza', 'pedestrian_safety': 'safe'},
        'PARK': {'category': 'park_road', 'pedestrian_safety': 'safe'},
        'WHARF': {'category': 'wharf', 'warnings': ['industrial_area']},
        'PIER': {'category': 'pier', 'warnings': ['water_access']},
    }
    
    if st_type in special_types:
        classification.update(special_types[st_type])
    
    # Zones industrielles ou d'accès restreint
    industrial_keywords = [
        'INDUSTRIAL', 'WAREHOUSE', 'TERMINAL', 'YARD', 'DEPOT',
        'PLANT', 'FACILITY', 'RESTRICTED', 'PRIVATE'
    ]
    
    for keyword in industrial_keywords:
        if keyword in street_name:
            classification.update({
                'category': 'industrial',
                'warnings': ['limited_access', 'industrial_traffic'],
                'pedestrian_safety': 'caution'
            })
            break
    
    return classification


def calculate_segment_cost(base_length, classification, time_of_day=None):
    """
    Calcule le coût d'un segment en tenant compte de sa classification
    
    Args:
        base_length: Longueur de base du segment
        classification: Classification du segment
        time_of_day: Heure de la journée (datetime)
        
    Returns:
        float: Coût ajusté du segment
    """
    # Facteur de base
    cost_factor = 1.0
    
    # Ajustements selon la catégorie
    category_factors = {
        'highway': float('inf'),  # Interdit
        'high_speed': float('inf'),  # Interdit
        'restricted_bridge': float('inf'),  # Interdit
        'ferry': 2.0,  # Coût élevé (temps d'attente)
        'transit': 1.5,  # Coût modéré
        'arterial': 1.3,  # Routes principales
        'bridge': 1.2,  # Ponts praticables
        'industrial': 1.4,  # Zones industrielles
        'alley': 1.1,  # Ruelles
        'plaza': 0.9,  # Zones piétonnes
        'park_road': 0.8,  # Routes de parc
        'normal': 1.0  # Routes normales
    }
    
    category = classification.get('category', 'normal')
    cost_factor *= category_factors.get(category, 1.0)
    
    # Ajustements selon l'heure pour certains types
    if time_of_day:
        hour = time_of_day.hour
        
        # Ferry: coûts variables selon les horaires
        if category == 'ferry':
            if 22 <= hour or hour <= 5:  # Service réduit la nuit
                cost_factor *= 3.0
            elif 7 <= hour <= 9 or 17 <= hour <= 19:  # Heures de pointe
                cost_factor *= 1.5
        
        # Transport public: ajustements selon les horaires
        elif category == 'transit':
            if 22 <= hour or hour <= 5:  # Service réduit
                cost_factor *= 2.0
            elif 7 <= hour <= 9 or 17 <= hour <= 19:  # Heures de pointe
                cost_factor *= 1.2
        
        # Routes industrielles: plus sûres pendant les heures de bureau
        elif category == 'industrial':
            if 9 <= hour <= 17:  # Heures de bureau
                cost_factor *= 0.9
            else:
                cost_factor *= 1.3
    
    return base_length * cost_factor


class ALTRouter:
    """
    Implémentation de l'algorithme ALT (A* avec Landmarks et Triangulation)
    pour le routage piéton évitant les zones à risque en fonction de l'heure.
    Inclut le filtrage des routes impraticables et la détection de transports spéciaux.
    """
    
    def __init__(self, risk_model=None):
        """
        Initialise le routeur ALT.
        
        Args:
            risk_model: Instance du modèle de risque contenant les données de rues et de risque
        """
        self.graph = None
        self.risk_model = risk_model
        self.landmarks = []
        self.landmark_distances = {}
        self.num_landmarks = 8  # Nombre de landmarks à sélectionner
        self.max_detour_factor = 1.5  # Facteur de détour maximum
        self.risk_weight = 0.7  # Importance du facteur de risque vs distance (0-1)
        self.allow_restricted_routes = False  # Permettre les routes avec transport requis
        self.pedestrian_only = True  # Mode piéton strict
    
    def build_graph(self, risk_df=None, streets_gdf=None):
        """
        Construit le graphe de routage à partir des données de rues et de risque,
        en filtrant les routes impraticables pour les piétons.
        
        Args:
            risk_df: GeoDataFrame contenant les données de risque
            streets_gdf: GeoDataFrame contenant les segments de rue
        """
        print("Construction du graphe de routage avec filtrage piéton...")
        
        # Utiliser les données du modèle de risque si aucune n'est fournie
        if risk_df is None and self.risk_model is not None and hasattr(self.risk_model, 'risk_df'):
            risk_df = self.risk_model.risk_df
            
        if streets_gdf is None and self.risk_model is not None and hasattr(self.risk_model, 'streets_gdf'):
            streets_gdf = self.risk_model.streets_gdf
        
        # Vérifier que les données sont disponibles
        if risk_df is None or streets_gdf is None:
            raise ValueError("Les données de rues et de risque sont requises pour construire le graphe")
        
        # Créer un graphe directionnel
        G = nx.DiGraph()
        
        # Statistiques de filtrage
        total_segments = 0
        walkable_segments = 0
        prohibited_segments = 0
        transport_required_segments = 0
        
        # Ajouter les nœuds (intersections) et les arêtes (segments de rue)
        street_segments = risk_df[risk_df['entity_type'] == 'street_segment']
        
        for idx, segment in street_segments.iterrows():
            total_segments += 1
            
            # Vérifier que la géométrie est valide et est une LineString
            if segment.geometry is None or not isinstance(segment.geometry, LineString):
                continue
            
            # Classifier le segment pour les piétons
            classification = classify_road_for_pedestrians(segment)
            
            # Filtrer selon le mode piéton
            if self.pedestrian_only and not classification['walkable']:
                prohibited_segments += 1
                continue
            
            # Compter les segments nécessitant un transport
            if classification['requires_transport']:
                transport_required_segments += 1
                if not self.allow_restricted_routes:
                    continue  # Ignorer si les transports spéciaux ne sont pas autorisés
            
            walkable_segments += 1
            
            # Obtenir les extrémités du segment
            start_point = Point(segment.geometry.coords[0])
            end_point = Point(segment.geometry.coords[-1])
            
            # Créer des identifiants pour les nœuds
            start_node = (round(start_point.x, 6), round(start_point.y, 6))
            end_node = (round(end_point.x, 6), round(end_point.y, 6))
            
            # Calculer la longueur du segment
            segment_length = segment.geometry.length
            
            # Obtenir les données de risque
            risk_score = segment.get('risk_score', 0)
            night_risk_factor = 1.0 + (segment.get('ped_crash_night_count', 0) / 
                                      max(segment.get('ped_crash_count', 1), 1))
            
            # Ajouter les nœuds avec leurs coordonnées
            if not G.has_node(start_node):
                G.add_node(start_node, x=start_point.x, y=start_point.y)
            
            if not G.has_node(end_node):
                G.add_node(end_node, x=end_point.x, y=end_point.y)
            
            # Ajouter l'arête avec les attributs étendus
            edge_attributes = {
                'id': idx,
                'length': segment_length,
                'base_risk': risk_score,
                'night_risk_factor': night_risk_factor,
                'street_name': segment.get('full_street_name', 'Rue inconnue'),
                'oneway': segment.get('ONEWAY', 'B') != 'TF',
                # Nouvelles informations piétonnes
                'pedestrian_classification': classification,
                'walkable': classification['walkable'],
                'requires_transport': classification['requires_transport'],
                'transport_type': classification.get('transport_type'),
                'safety_level': classification['pedestrian_safety'],
                'warnings': classification.get('warnings', []),
                'restrictions': classification.get('restrictions', [])
            }
            
            G.add_edge(start_node, end_node, **edge_attributes)
            
            # Si la rue est à double sens, ajouter l'arête inverse
            if segment.get('ONEWAY', 'B') not in ['FT', 'TF']:
                G.add_edge(end_node, start_node, **edge_attributes)
        
        self.graph = G
        
        # Afficher les statistiques de filtrage
        print(f"\n📊 Statistiques du filtrage piéton :")
        print(f"  • Segments totaux analysés: {total_segments:,}")
        print(f"  • Segments praticables: {walkable_segments:,} ({walkable_segments/total_segments*100:.1f}%)")
        print(f"  • Segments interdits: {prohibited_segments:,} ({prohibited_segments/total_segments*100:.1f}%)")
        print(f"  • Segments nécessitant transport: {transport_required_segments:,}")
        print(f"  • Graphe final: {len(G.nodes):,} nœuds, {len(G.edges):,} arêtes")
        
        # Sélectionner les landmarks et précalculer les distances
        self.select_landmarks_optimized()
        
        return G
    
    def select_landmarks_optimized(self):
        """
        Version optimisée de la sélection de landmarks
        """
        if self.graph is None or len(self.graph.nodes) == 0:
            print("Le graphe doit être construit avant de sélectionner les landmarks")
            return
        
        start_time = time.time()
        
        print(f"Sélection optimisée de {self.num_landmarks} landmarks...")
        print(f"Graphe : {len(self.graph.nodes)} nœuds, {len(self.graph.edges)} arêtes")
        
        # Récupérer la liste des nœuds
        nodes = list(self.graph.nodes)
        
        # Sélectionner le premier landmark aléatoirement
        first_landmark = random.choice(nodes)
        landmarks = [first_landmark]
        
        print(f"Landmark 1/{self.num_landmarks} sélectionné: {first_landmark}")
        
        # Pour chaque landmark suivant
        for landmark_num in range(2, self.num_landmarks + 1):
            print(f"Sélection du landmark {landmark_num}/{self.num_landmarks}...")
            step_start = time.time()
            
            # Dictionnaire pour stocker la distance minimale
            min_distances_to_landmarks = {}
            
            # Pour chaque landmark existant, calculer les distances vers TOUS les nœuds
            for landmark in landmarks:
                try:
                    distances_from_landmark = nx.single_source_dijkstra_path_length(
                        self.graph, landmark, weight='length'
                    )
                    
                    # Mettre à jour la distance minimale pour chaque nœud
                    for node, distance in distances_from_landmark.items():
                        if node not in min_distances_to_landmarks:
                            min_distances_to_landmarks[node] = distance
                        else:
                            min_distances_to_landmarks[node] = min(
                                min_distances_to_landmarks[node], 
                                distance
                            )
                            
                except nx.NetworkXNoPath:
                    print(f"Avertissement: Landmark {landmark} non connecté à certains nœuds")
                    continue
            
            # Trouver le nœud avec la distance minimale maximale (farthest-first)
            best_candidate = None
            max_min_distance = -1
            
            for node, min_distance in min_distances_to_landmarks.items():
                if node in landmarks:
                    continue
                    
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = node
            
            if best_candidate is None:
                print(f"Impossible de trouver un landmark {landmark_num}, arrêt à {len(landmarks)} landmarks")
                break
                
            landmarks.append(best_candidate)
            
            step_time = time.time() - step_start
            print(f"Landmark {landmark_num}/{self.num_landmarks} sélectionné: {best_candidate} "
                  f"(distance min: {convert_degrees_to_miles(max_min_distance):.2f} miles) en {step_time:.2f}s")
        
        self.landmarks = landmarks
        total_time = time.time() - start_time
        print(f"\n🎉 Sélection terminée en {total_time:.2f} secondes !")
        
        # Précalculer les distances depuis chaque landmark
        print("\n🔄 Précalcul des distances pour l'optimisation ALT...")
        precalc_start = time.time()
        
        self.landmark_distances = {}
        
        for i, landmark in enumerate(self.landmarks, 1):
            print(f"Précalcul depuis landmark {i}/{len(self.landmarks)}...")
            try:
                distances = nx.single_source_dijkstra_path_length(self.graph, landmark, weight='length')
                self.landmark_distances[landmark] = distances
            except nx.NetworkXNoPath:
                print(f"Impossible de calculer les distances depuis le landmark {landmark}")
        
        precalc_time = time.time() - precalc_start
        total_time_with_precalc = time.time() - start_time
        
        print(f"Précalcul terminé en {precalc_time:.2f} secondes")
        print(f"⚡ Temps total: {total_time_with_precalc:.2f} secondes")
        
        return landmarks
    
    def calculate_risk_weight(self, edge_data, time=None):
        """
        Calcule le poids de risque pour un segment en fonction de l'heure et de sa classification.
        """
        base_risk = edge_data.get('base_risk', 0)
        night_risk_factor = edge_data.get('night_risk_factor', 1.0)
        classification = edge_data.get('pedestrian_classification', {})
        
        # Ajuster le risque selon la classification piétonne
        safety_multipliers = {
            'safe': 1.0,
            'caution': 1.3,
            'prohibited': float('inf'),
            'safe_with_transport': 1.1
        }
        
        safety_level = classification.get('pedestrian_safety', 'safe')
        safety_multiplier = safety_multipliers.get(safety_level, 1.0)
        
        # Ajuster le risque en fonction de l'heure
        if time is not None:
            hour = time.hour
            
            # Augmenter le risque la nuit
            if hour >= 20 or hour < 6:
                risk = base_risk * night_risk_factor * safety_multiplier
                
                # Risque supplémentaire pour certains types de routes la nuit
                if 'limited_lighting' in classification.get('warnings', []):
                    risk *= 1.5
                if classification.get('category') == 'industrial':
                    risk *= 1.4
                    
            # Heures de pointe
            elif (hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 18):
                risk = base_risk * 1.2 * safety_multiplier
                
                # Risque réduit sur les routes avec beaucoup de trafic (plus de surveillance)
                if classification.get('category') in ['arterial', 'bridge']:
                    risk *= 0.9
            else:
                risk = base_risk * safety_multiplier
        else:
            risk = base_risk * safety_multiplier
        
        return risk
    
    def calculate_edge_weight(self, u, v, edge_data, time=None):
        """
        Calcule le poids total d'une arête en tenant compte de la classification piétonne.
        """
        # Obtenir la longueur de base
        base_length = edge_data.get('length', 1.0)
        
        # Obtenir la classification
        classification = edge_data.get('pedestrian_classification', {})
        
        # Calculer le coût basé sur la classification
        segment_cost = calculate_segment_cost(base_length, classification, time)
        
        # Ajouter le facteur de risque
        risk = self.calculate_risk_weight(edge_data, time)
        
        # Combiner coût de base et risque
        total_weight = segment_cost * (1 + self.risk_weight * risk)
        
        return total_weight
    
    def landmark_heuristic(self, node, target):
        """
        Calcule l'heuristique basée sur les landmarks pour l'algorithme ALT.
        """
        max_h = 0
        
        for landmark in self.landmarks:
            if landmark not in self.landmark_distances:
                continue
                
            landmark_dists = self.landmark_distances[landmark]
            
            if node not in landmark_dists or target not in landmark_dists:
                continue
            
            # Utiliser l'inégalité triangulaire
            h1 = landmark_dists[target] - landmark_dists[node]
            h2 = landmark_dists[node] - landmark_dists[target]
            
            h = max(h1, h2)
            max_h = max(max_h, h)
        
        return max(0, max_h)
    
    def a_star_search(self, start_point, end_point, time=None):
        """
        Implémentation de l'algorithme ALT avec prise en compte des classifications piétonnes.
        """
        if self.graph is None:
            raise ValueError("Le graphe doit être construit avant de calculer un itinéraire")
            
        # Trouver les nœuds les plus proches
        start_node = self.find_nearest_node(start_point)
        end_node = self.find_nearest_node(end_point)
        
        if start_node is None or end_node is None:
            raise ValueError("Impossible de trouver un nœud proche des points spécifiés")
            
        print(f"Routage de {start_node} à {end_node}")
        
        # File de priorité pour A*
        open_set = [(0, start_node)]
        heapq.heapify(open_set)
        
        # Pour reconstruire le chemin
        came_from = {}
        
        # Coût du début jusqu'au nœud
        g_score = {node: float('inf') for node in self.graph.nodes}
        g_score[start_node] = 0
        
        # Coût estimé
        f_score = {node: float('inf') for node in self.graph.nodes}
        f_score[start_node] = self.landmark_heuristic(start_node, end_node)
        
        # Ensemble des nœuds déjà évalués
        closed_set = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            # Si on a atteint la destination
            if current == end_node:
                # Reconstruire le chemin
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # Calculer les métriques
                total_distance = 0
                total_risk = 0
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = self.graph.get_edge_data(u, v)
                    total_distance += edge_data.get('length', 0)
                    total_risk += self.calculate_risk_weight(edge_data, time)
                
                avg_risk = total_risk / (len(path) - 1) if len(path) > 1 else 0
                
                return path, total_distance, avg_risk
            
            # Marquer comme visité
            closed_set.add(current)
            
            # Évaluer les voisins
            for neighbor in self.graph.neighbors(current):
                if neighbor in closed_set:
                    continue
                
                edge_data = self.graph.get_edge_data(current, neighbor)
                
                # Calculer le nouveau score g
                new_g_score = g_score[current] + self.calculate_edge_weight(current, neighbor, edge_data, time)
                
                # Si nous avons trouvé un meilleur chemin
                if new_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = new_g_score
                    new_f_score = new_g_score + self.landmark_heuristic(neighbor, end_node)
                    f_score[neighbor] = new_f_score
                    
                    if neighbor not in [node for _, node in open_set]:
                        heapq.heappush(open_set, (new_f_score, neighbor))
        
        # Aucun chemin trouvé
        return None, 0, 0
    
    def find_nearest_node(self, point):
        """
        Trouve le nœud du graphe le plus proche d'un point donné.
        """
        if self.graph is None or len(self.graph.nodes) == 0:
            return None
            
        # Extraire les coordonnées
        if isinstance(point, Point):
            x, y = point.x, point.y
        else:
            x, y = point
            
        # Calculer la distance euclidienne
        min_dist = float('inf')
        nearest_node = None
        
        for node in self.graph.nodes:
            node_x = self.graph.nodes[node]['x']
            node_y = self.graph.nodes[node]['y']
            
            dist = ((node_x - x) ** 2 + (node_y - y) ** 2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                
        return nearest_node
    
    def get_safe_route(self, start_point, end_point, time=None):
        """
        Calcule un itinéraire sécurisé avec informations détaillées sur les transports requis.
        """
        if time is None:
            time = datetime.now()
            
        # Calculer l'itinéraire
        path, distance_degrees, avg_risk = self.a_star_search(start_point, end_point, time)
        
        if path is None:
            return {
                "status": "error",
                "message": "Aucun itinéraire trouvé"
            }
        
        # Conversions
        distance_miles = convert_degrees_to_miles(distance_degrees)
        distance_meters = convert_degrees_to_meters(distance_degrees)
        
        # Calcul du temps
        walking_speed_ms = 1.4  # m/s
        estimated_time_seconds = distance_meters / walking_speed_ms
        time_info = format_walking_time(estimated_time_seconds)
        
        # Analyser les segments pour les transports requis et avertissements
        segments = []
        transport_required = False
        transport_segments = []
        warnings = set()
        route_warnings = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.graph.get_edge_data(u, v)
            
            # Informations de base
            street_name = edge_data.get('street_name', 'Rue inconnue')
            length_degrees = edge_data.get('length', 0)
            length_miles = convert_degrees_to_miles(length_degrees)
            risk = self.calculate_risk_weight(edge_data, time)
            
            # Informations piétonnes
            classification = edge_data.get('pedestrian_classification', {})
            requires_transport = edge_data.get('requires_transport', False)
            transport_type = edge_data.get('transport_type')
            segment_warnings = edge_data.get('warnings', [])
            safety_level = edge_data.get('safety_level', 'safe')
            
            # Collecter les transports requis
            if requires_transport:
                transport_required = True
                transport_segments.append({
                    'segment_index': i,
                    'street': street_name,
                    'transport_type': transport_type,
                    'length_miles': round(length_miles, 4)
                })
            
            # Collecter les avertissements
            warnings.update(segment_warnings)
            
            # Segment détaillé
            segment_info = {
                "street": street_name,
                "length_degrees": length_degrees,
                "length_miles": round(length_miles, 4),
                "risk": risk,
                "coordinates": [u, v],
                "pedestrian_info": {
                    "walkable": edge_data.get('walkable', True),
                    "safety_level": safety_level,
                    "category": classification.get('category', 'normal'),
                    "requires_transport": requires_transport,
                    "transport_type": transport_type,
                    "warnings": segment_warnings
                }
            }
            
            segments.append(segment_info)
        
        # Convertir les avertissements en messages
        warning_messages = {
            'high_traffic': 'Trafic dense - soyez vigilant',
            'limited_sidewalks': 'Trottoirs limités ou absents',
            'wind_exposure': 'Exposition au vent (pont/zone ouverte)',
            'weather_dependent': 'Conditions dépendantes de la météo',
            'schedule_dependent': 'Horaires de transport à vérifier',
            'requires_ticket': 'Ticket/paiement requis',
            'industrial_area': 'Zone industrielle - circulation de poids lourds',
            'limited_lighting': 'Éclairage limité',
            'narrow': 'Passage étroit',
            'dead_end': 'Impasse possible'
        }
        
        route_warnings = [warning_messages.get(w, w) for w in warnings if w in warning_messages]
        
        # Déterminer le niveau de risque
        if avg_risk >= 0.75:
            risk_level = "Très élevé"
        elif avg_risk >= 0.5:
            risk_level = "Élevé"
        elif avg_risk >= 0.25:
            risk_level = "Modéré"
        else:
            risk_level = "Faible"
        
        # Réponse complète
        route = {
            "status": "success",
            "route": {
                "distance": {
                    "miles": round(distance_miles, 2),
                    "kilometers": round(distance_miles * 1.60934, 2),
                    "raw_degrees": distance_degrees
                },
                "estimated_time": time_info,
                "walking_speed": {
                    "mph": round(walking_speed_ms * 2.237, 1),
                    "kmh": round(walking_speed_ms * 3.6, 1)
                },
                "risk_score": avg_risk,
                "risk_level": risk_level,
                "transport_required": transport_required,
                "transport_segments": transport_segments,
                "route_warnings": route_warnings,
                "segments": segments,
                "summary": {
                    "total_segments": len(segments),
                    "walkable_segments": sum(1 for s in segments if s['pedestrian_info']['walkable']),
                    "transport_required_segments": len(transport_segments),
                    "high_risk_segments": sum(1 for s in segments if s['risk'] > 0.05),
                    "average_segment_length_miles": round(distance_miles / len(segments), 4) if segments else 0
                }
            },
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return route
    
    def get_route_as_geojson(self, start_point, end_point, time=None):
        """
        Génère un GeoJSON avec informations sur les transports requis.
        """
        route_info = self.get_safe_route(start_point, end_point, time)
        
        if route_info["status"] == "error":
            return {
                "type": "FeatureCollection",
                "features": []
            }
            
        route = route_info["route"]
        segments = route["segments"]
        
        features = []
        
        # Ajouter une feature pour chaque segment
        for i, segment in enumerate(segments):
            coords = segment["coordinates"]
            ped_info = segment["pedestrian_info"]
            
            # Couleur selon le type de segment
            if ped_info["requires_transport"]:
                color = "#9932CC"  # Violet pour transport requis
                stroke_width = 4
            elif ped_info["safety_level"] == "caution":
                color = "#FFA500"  # Orange pour prudence
                stroke_width = 3
            elif segment["risk"] > 0.05:
                color = "#FF0000"  # Rouge pour haut risque
                stroke_width = 3
            elif segment["risk"] > 0.01:
                color = "#FFFF00"  # Jaune pour risque modéré
                stroke_width = 2
            else:
                color = "#00FF00"  # Vert pour sécurisé
                stroke_width = 2
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(coords[0]), list(coords[1])]
                },
                "properties": {
                    "segment_id": i,
                    "street": segment["street"],
                    "length_miles": segment["length_miles"],
                    "risk": segment["risk"],
                    "walkable": ped_info["walkable"],
                    "safety_level": ped_info["safety_level"],
                    "requires_transport": ped_info["requires_transport"],
                    "transport_type": ped_info["transport_type"],
                    "warnings": ped_info["warnings"],
                    "color": color,
                    "stroke": color,
                    "stroke-width": stroke_width,
                    "stroke-opacity": 0.8
                }
            }
            
            features.append(feature)
        
        # Points de départ et d'arrivée
        start_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": list(start_point) if isinstance(start_point, tuple) else [start_point.x, start_point.y]
            },
            "properties": {
                "type": "start",
                "title": "Départ",
                "marker-color": "#00FF00",
                "marker-size": "large"
            }
        }
        
        end_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": list(end_point) if isinstance(end_point, tuple) else [end_point.x, end_point.y]
            },
            "properties": {
                "type": "end",
                "title": "Arrivée",
                "marker-color": "#FF0000",
                "marker-size": "large"
            }
        }
        
        features.extend([start_feature, end_feature])
        
        # GeoJSON final
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "route_info": {
                    "distance_miles": route["distance"]["miles"],
                    "estimated_time": route["estimated_time"]["formatted"],
                    "transport_required": route["transport_required"],
                    "transport_segments": route["transport_segments"],
                    "warnings": route["route_warnings"],
                    "risk_level": route["risk_level"]
                }
            }
        }
        
        return geojson
    
    def set_pedestrian_mode(self, strict_mode=True, allow_transports=False):
        """
        Configure le mode piéton du routeur.
        
        Args:
            strict_mode: Si True, exclut toutes les routes non praticables
            allow_transports: Si True, autorise les segments nécessitant un transport
        """
        self.pedestrian_only = strict_mode
        self.allow_restricted_routes = allow_transports
        
        print(f"Mode piéton configuré:")
        print(f"  • Mode strict: {strict_mode}")
        print(f"  • Transports autorisés: {allow_transports}")
    
    def set_risk_weight(self, weight):
        """
        Définit l'importance du facteur de risque par rapport à la distance.
        """
        if 0 <= weight <= 1:
            self.risk_weight = weight
        else:
            raise ValueError("Le poids du risque doit être entre 0 et 1")
    
    def set_max_detour_factor(self, factor):
        """
        Définit le facteur de détour maximum autorisé.
        """
        if factor >= 1.0:
            self.max_detour_factor = factor
        else:
            raise ValueError("Le facteur de détour doit être supérieur ou égal à 1.0")