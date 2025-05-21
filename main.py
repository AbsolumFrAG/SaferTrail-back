import argparse
import json
from src.risk_model import PedestrianRiskModel
from src.api import start_api

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Modèle pour analyser la sécurité des piétons à Boston')
    
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Commande 'train'
    train_parser = subparsers.add_parser('train', help='Entraîner le modèle')
    train_parser.add_argument('--force', action='store_true', help='Force le réentraînement du modèle')
    
    # Commande 'risk-zones'
    zones_parser = subparsers.add_parser('risk-zones', help='Obtenir les zones à risque pour les piétons')
    zones_parser.add_argument('--threshold', type=float, default=0.7, help='Seuil de score de risque')
    zones_parser.add_argument('--limit', type=int, default=10, help='Nombre maximum de zones à retourner')
    zones_parser.add_argument('--output', type=str, default='risk_zones.json', help='Fichier de sortie')
    
    # Commande 'clusters'
    clusters_parser = subparsers.add_parser('clusters', help='Obtenir les clusters de zones à risque')
    clusters_parser.add_argument('--min-size', type=int, default=2, help='Taille minimale d\'un cluster')
    clusters_parser.add_argument('--output', type=str, default='risk_clusters.json', help='Fichier de sortie')
    
    # Commande 'recommendations'
    recommendations_parser = subparsers.add_parser('recommendations', help='Obtenir des recommandations de sécurité')
    recommendations_parser.add_argument('zone_id', type=int, help='Identifiant de la zone')
    recommendations_parser.add_argument('--output', type=str, help='Fichier de sortie (optionnel)')
    
    # Commande 'statistics'
    statistics_parser = subparsers.add_parser('statistics', help='Obtenir des statistiques globales')
    statistics_parser.add_argument('--output', type=str, default='statistics.json', help='Fichier de sortie')
    
    # Commande 'api'
    api_parser = subparsers.add_parser('api', help='Démarrer l\'API')
    api_parser.add_argument('--host', type=str, default='0.0.0.0', help='Hôte de l\'API')
    api_parser.add_argument('--port', type=int, default=8000, help='Port de l\'API')
    api_parser.add_argument('--debug', action='store_true', help='Mode debug')
    
    return parser.parse_args()

def main():
    """Point d'entrée principal"""
    args = parse_args()
    
    # Initialiser le modèle
    model = PedestrianRiskModel()
    
    if args.command == 'train':
        # Entraîner le modèle
        print(f"Entraînement du modèle (force={args.force})...")
        model.train(force_retrain=args.force)
        print("Entraînement terminé.")
    
    elif args.command == 'risk-zones':
        # Obtenir les zones à risque pour les piétons
        print(f"Obtention des zones à risque pour les piétons (threshold={args.threshold}, limit={args.limit})...")
        risk_zones = model.get_high_risk_zones(threshold=args.threshold, limit=args.limit)
        
        # Sauvegarder les résultats
        with open(args.output, 'w') as f:
            json.dump(risk_zones, f, indent=2)
        
        print(f"{len(risk_zones)} zones à risque trouvées et sauvegardées dans {args.output}.")
    
    elif args.command == 'clusters':
        # Obtenir les clusters de zones à risque
        print(f"Obtention des clusters de zones à risque (min_size={args.min_size})...")
        clusters = model.get_risk_clusters(min_cluster_size=args.min_size)
        
        # Sauvegarder les résultats
        with open(args.output, 'w') as f:
            json.dump(clusters, f, indent=2)
        
        print(f"{len(clusters)} clusters trouvés et sauvegardés dans {args.output}.")
    
    elif args.command == 'recommendations':
        # Obtenir des recommandations de sécurité
        print(f"Obtention des recommandations de sécurité pour la zone {args.zone_id}...")
        recommendations = model.get_safety_recommendations(args.zone_id)
        
        # Afficher ou sauvegarder les résultats
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(recommendations, f, indent=2)
            print(f"Recommandations sauvegardées dans {args.output}.")
        else:
            print(json.dumps(recommendations, indent=2))
    
    elif args.command == 'statistics':
        # Charger le modèle si nécessaire
        if model.risk_df is None:
            model.load_model()
        
        # Calculer des statistiques globales
        if model.risk_df is None:
            print("Erreur: Aucun modèle chargé.")
            return
        
        # Nombre total d'accidents de piétons
        total_crashes = model.risk_df['ped_crash_count'].sum()
        
        # Nombre total d'accidents de piétons de nuit
        total_night_crashes = model.risk_df['ped_crash_night_count'].sum()
        
        # Nombre total de signalements liés aux piétons
        total_reports = model.risk_df['ped_311_count'].sum()
        
        # Nombre de zones à risque élevé (score > 0.7)
        high_risk_zones = len(model.risk_df[model.risk_df['risk_score'] >= 0.7])
        
        # Problèmes les plus courants
        sidewalk_issues = model.risk_df['sidewalk_issues'].sum()
        crossing_issues = model.risk_df['crossing_issues'].sum()
        lighting_issues = model.risk_df['lighting_issues'].sum()
        
        statistics = {
            'total_pedestrian_crashes': int(total_crashes),
            'total_night_crashes': int(total_night_crashes),
            'total_pedestrian_related_reports': int(total_reports),
            'high_risk_zones_count': int(high_risk_zones),
            'common_issues': {
                'sidewalk_issues': int(sidewalk_issues),
                'crossing_issues': int(crossing_issues),
                'lighting_issues': int(lighting_issues)
            }
        }
        
        # Sauvegarder les résultats
        with open(args.output, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"Statistiques sauvegardées dans {args.output}.")
    
    elif args.command == 'api':
        # Démarrer l'API
        print(f"Démarrage de l'API sur {args.host}:{args.port}...")
        start_api(host=args.host, port=args.port, debug=args.debug)
    
    else:
        print("Commande non reconnue. Utilisez --help pour voir les commandes disponibles.")

if __name__ == '__main__':
    main()