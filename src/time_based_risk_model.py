import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

class TimeBasedPedestrianRiskModel:
    """
    Modèle temporel pour l'analyse et la prédiction du risque pour les piétons en fonction de l'heure
    """
    
    def __init__(self):
        """Initialisation du modèle de risque temporel pour les piétons"""
        self.time_risk_data = None  # Données temporelles agrégées
        self.hourly_risk_scores = None  # Scores de risque par heure
        self.daily_risk_scores = None  # Scores de risque par jour de la semaine
        self.monthly_risk_scores = None  # Scores de risque par mois
        self.prediction_model = None  # Modèle ML pour prédiction avancée
        self.model_path = Path(__file__).parent.parent / 'models' / 'temporal_risk_model.pkl'
        
        # Définir les périodes de la journée
        self.time_periods = {
            'nuit': [0, 1, 2, 3, 4, 5],
            'matin': [6, 7, 8, 9, 10, 11],
            'après-midi': [12, 13, 14, 15, 16, 17],
            'soir': [18, 19, 20, 21, 22, 23]
        }
        
    def analyze_temporal_patterns(self, df_crashes, df_311=None, df_crimes=None):
        """
        Analyse les patterns temporels dans les données d'accidents et d'incidents
        
        Args:
            df_crashes (DataFrame): Données d'accidents impliquant des piétons
            df_311 (DataFrame, optional): Données 311 liées aux piétons
            df_crimes (DataFrame, optional): Données de crimes liés à la sécurité des piétons
            
        Returns:
            dict: Dictionnaire contenant les analyses temporelles
        """
        print("Analyse des patterns temporels...")
        
        # Initialiser les dictionnaires pour stocker les analyses temporelles
        hourly_counts = {'accidents': {}, '311_reports': {}, 'crimes': {}}
        daily_counts = {'accidents': {}, '311_reports': {}, 'crimes': {}}
        monthly_counts = {'accidents': {}, '311_reports': {}, 'crimes': {}}
        
        # Analyse des accidents par heure, jour et mois
        if df_crashes is not None and not df_crashes.empty:
            # S'assurer que les colonnes temporelles existent
            if 'dispatch_ts' in df_crashes.columns:
                # Extraire l'heure, le jour de la semaine et le mois
                if 'hour' not in df_crashes.columns:
                    df_crashes['hour'] = df_crashes['dispatch_ts'].dt.hour
                if 'day_of_week' not in df_crashes.columns:
                    df_crashes['day_of_week'] = df_crashes['dispatch_ts'].dt.day_name()
                if 'month' not in df_crashes.columns:
                    df_crashes['month'] = df_crashes['dispatch_ts'].dt.month
                
                # Compter les accidents par heure
                hourly_counts['accidents'] = df_crashes['hour'].value_counts().sort_index().to_dict()
                
                # Compter les accidents par jour de la semaine
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = df_crashes['day_of_week'].value_counts().reindex(day_order).fillna(0).to_dict()
                daily_counts['accidents'] = day_counts
                
                # Compter les accidents par mois
                monthly_counts['accidents'] = df_crashes['month'].value_counts().sort_index().to_dict()
        
        # Analyse des signalements 311 par heure, jour et mois
        if df_311 is not None and not df_311.empty:
            # S'assurer que les colonnes temporelles existent
            if 'open_dt' in df_311.columns:
                # Extraire l'heure, le jour de la semaine et le mois
                if 'hour' not in df_311.columns:
                    df_311['hour'] = df_311['open_dt'].dt.hour
                if 'day_of_week' not in df_311.columns:
                    df_311['day_of_week'] = df_311['open_dt'].dt.day_name()
                if 'month' not in df_311.columns:
                    df_311['month'] = df_311['open_dt'].dt.month
                
                # Compter les signalements par heure
                hourly_counts['311_reports'] = df_311['hour'].value_counts().sort_index().to_dict()
                
                # Compter les signalements par jour de la semaine
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = df_311['day_of_week'].value_counts().reindex(day_order).fillna(0).to_dict()
                daily_counts['311_reports'] = day_counts
                
                # Compter les signalements par mois
                monthly_counts['311_reports'] = df_311['month'].value_counts().sort_index().to_dict()
        
        # Analyse des crimes par heure, jour et mois
        if df_crimes is not None and not df_crimes.empty:
            # S'assurer que les colonnes temporelles existent
            if 'OCCURRED_ON_DATE' in df_crimes.columns:
                # Extraire l'heure, le jour de la semaine et le mois
                if 'hour' not in df_crimes.columns:
                    df_crimes['hour'] = df_crimes['OCCURRED_ON_DATE'].dt.hour
                if 'day_of_week' not in df_crimes.columns:
                    df_crimes['day_of_week'] = df_crimes['OCCURRED_ON_DATE'].dt.day_name()
                if 'month' not in df_crimes.columns:
                    df_crimes['month'] = df_crimes['OCCURRED_ON_DATE'].dt.month
                
                # Compter les crimes par heure
                hourly_counts['crimes'] = df_crimes['hour'].value_counts().sort_index().to_dict()
                
                # Compter les crimes par jour de la semaine
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = df_crimes['day_of_week'].value_counts().reindex(day_order).fillna(0).to_dict()
                daily_counts['crimes'] = day_counts
                
                # Compter les crimes par mois
                monthly_counts['crimes'] = df_crimes['month'].value_counts().sort_index().to_dict()
        
        # Calculer les scores de risque normalisés
        self.calculate_temporal_risk_scores(hourly_counts, daily_counts, monthly_counts)
        
        # Stocker les données temporelles agrégées
        self.time_risk_data = {
            'hourly_counts': hourly_counts,
            'daily_counts': daily_counts,
            'monthly_counts': monthly_counts,
            'hourly_risk_scores': self.hourly_risk_scores,
            'daily_risk_scores': self.daily_risk_scores,
            'monthly_risk_scores': self.monthly_risk_scores
        }
        
        # Entraîner un modèle prédictif plus avancé si suffisamment de données
        self.train_prediction_model(df_crashes, df_311, df_crimes)
        
        return self.time_risk_data
    
    def calculate_temporal_risk_scores(self, hourly_counts, daily_counts, monthly_counts):
        """
        Calcule les scores de risque temporels normalisés
        
        Args:
            hourly_counts (dict): Nombre d'incidents par heure
            daily_counts (dict): Nombre d'incidents par jour de la semaine
            monthly_counts (dict): Nombre d'incidents par mois
        """
        # Initialiser les dictionnaires pour les scores de risque
        hourly_risk = {}
        daily_risk = {}
        monthly_risk = {}
        
        # Calculer les scores de risque horaires
        for hour in range(24):
            score = 0
            # Pondérer les différentes sources de données
            accident_weight = 0.6
            report_weight = 0.3
            crime_weight = 0.1
            
            # Accidents (poids plus élevé)
            if 'accidents' in hourly_counts and hourly_counts['accidents'] and hour in hourly_counts['accidents']:
                score += hourly_counts['accidents'][hour] * accident_weight
            
            # Signalements 311
            if '311_reports' in hourly_counts and hourly_counts['311_reports'] and hour in hourly_counts['311_reports']:
                score += hourly_counts['311_reports'][hour] * report_weight
            
            # Crimes
            if 'crimes' in hourly_counts and hourly_counts['crimes'] and hour in hourly_counts['crimes']:
                score += hourly_counts['crimes'][hour] * crime_weight
            
            hourly_risk[hour] = score
        
        # Normaliser les scores horaires entre 0 et 1
        max_hourly_score = max(hourly_risk.values()) if hourly_risk.values() else 1
        for hour in hourly_risk:
            hourly_risk[hour] = hourly_risk[hour] / max_hourly_score if max_hourly_score > 0 else 0
        
        # Calculer les scores de risque par jour de la semaine
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in day_order:
            score = 0
            
            # Accidents
            if 'accidents' in daily_counts and daily_counts['accidents'] and day in daily_counts['accidents']:
                score += daily_counts['accidents'][day] * accident_weight
            
            # Signalements 311
            if '311_reports' in daily_counts and daily_counts['311_reports'] and day in daily_counts['311_reports']:
                score += daily_counts['311_reports'][day] * report_weight
            
            # Crimes
            if 'crimes' in daily_counts and daily_counts['crimes'] and day in daily_counts['crimes']:
                score += daily_counts['crimes'][day] * crime_weight
            
            daily_risk[day] = score
        
        # Normaliser les scores journaliers entre 0 et 1
        max_daily_score = max(daily_risk.values()) if daily_risk.values() else 1
        for day in daily_risk:
            daily_risk[day] = daily_risk[day] / max_daily_score if max_daily_score > 0 else 0
        
        # Calculer les scores de risque par mois
        for month in range(1, 13):
            score = 0
            
            # Accidents
            if 'accidents' in monthly_counts and monthly_counts['accidents'] and month in monthly_counts['accidents']:
                score += monthly_counts['accidents'][month] * accident_weight
            
            # Signalements 311
            if '311_reports' in monthly_counts and monthly_counts['311_reports'] and month in monthly_counts['311_reports']:
                score += monthly_counts['311_reports'][month] * report_weight
            
            # Crimes
            if 'crimes' in monthly_counts and monthly_counts['crimes'] and month in monthly_counts['crimes']:
                score += monthly_counts['crimes'][month] * crime_weight
            
            monthly_risk[month] = score
        
        # Normaliser les scores mensuels entre 0 et 1
        max_monthly_score = max(monthly_risk.values()) if monthly_risk.values() else 1
        for month in monthly_risk:
            monthly_risk[month] = monthly_risk[month] / max_monthly_score if max_monthly_score > 0 else 0
        
        # Stocker les scores de risque
        self.hourly_risk_scores = hourly_risk
        self.daily_risk_scores = daily_risk
        self.monthly_risk_scores = monthly_risk
    
    def train_prediction_model(self, df_crashes, df_311=None, df_crimes=None):
        """
        Entraîne un modèle prédictif avancé pour estimer le risque à un moment donné
        
        Args:
            df_crashes (DataFrame): Données d'accidents impliquant des piétons
            df_311 (DataFrame, optional): Données 311 liées aux piétons
            df_crimes (DataFrame, optional): Données de crimes liés à la sécurité des piétons
        """
        print("Entraînement du modèle prédictif temporel...")
        
        # Créer un DataFrame combiné pour l'entraînement
        combined_data = []
        
        # Ajouter les accidents
        if df_crashes is not None and not df_crashes.empty and 'dispatch_ts' in df_crashes.columns:
            df = df_crashes.copy()
            df['event_type'] = 'accident'
            df['timestamp'] = df['dispatch_ts']
            # S'assurer que timestamp est bien en datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['severity'] = 1.0  # Accidents ont un poids plus élevé
            combined_data.append(df[['timestamp', 'event_type', 'severity']])
        
        # Ajouter les signalements 311
        if df_311 is not None and not df_311.empty and 'open_dt' in df_311.columns:
            df = df_311.copy()
            df['event_type'] = '311_report'
            df['timestamp'] = df['open_dt']
            # S'assurer que timestamp est bien en datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['severity'] = 0.5  # Signalements ont un poids modéré
            combined_data.append(df[['timestamp', 'event_type', 'severity']])
        
        # Ajouter les crimes
        if df_crimes is not None and not df_crimes.empty and 'OCCURRED_ON_DATE' in df_crimes.columns:
            df = df_crimes.copy()
            df['event_type'] = 'crime'
            df['timestamp'] = df['OCCURRED_ON_DATE']
            # S'assurer que timestamp est bien en datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['severity'] = 0.8  # Crimes ont un poids élevé
            combined_data.append(df[['timestamp', 'event_type', 'severity']])
        
        # Fusionner les données si disponibles
        if combined_data:
            df_combined = pd.concat(combined_data, ignore_index=True)
            
            # S'assurer que timestamp est bien un objet datetime après la concaténation
            df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'], errors='coerce')
            
            # Filtrer les lignes avec des dates valides
            df_combined = df_combined.dropna(subset=['timestamp'])
            
            if df_combined.empty:
                print("Après nettoyage des dates invalides, aucune donnée temporelle valide n'est disponible.")
                self.prediction_model = None
                self.prediction_grid = None
                return
            
            # Extraire les caractéristiques temporelles
            df_combined['hour'] = df_combined['timestamp'].dt.hour
            df_combined['day_of_week'] = df_combined['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
            df_combined['month'] = df_combined['timestamp'].dt.month
            df_combined['is_weekend'] = df_combined['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            df_combined['is_rush_hour'] = df_combined['hour'].apply(
                lambda x: 1 if (x >= 7 and x <= 9) or (x >= 16 and x <= 18) else 0
            )
            df_combined['is_night'] = df_combined['hour'].apply(lambda x: 1 if x >= 20 or x <= 5 else 0)
            
            # Créer un tableau de contingence par heure et jour de la semaine
            pivot_table = pd.pivot_table(
                df_combined, 
                values='severity', 
                index=['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour', 'is_night'],
                aggfunc='count'
            ).reset_index()
            
            # Si suffisamment de données, entraîner un modèle de Random Forest
            if len(pivot_table) > 10:
                try:
                    # Préparer les données d'entraînement
                    X = pivot_table[['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour', 'is_night']]
                    y = pivot_table['severity']
                    
                    # Diviser en ensembles d'entraînement et de test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Entraîner un modèle Random Forest
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    
                    # Évaluer le modèle
                    train_score = rf_model.score(X_train, y_train)
                    test_score = rf_model.score(X_test, y_test)
                    print(f"Score du modèle (train): {train_score:.4f}")
                    print(f"Score du modèle (test): {test_score:.4f}")
                    
                    # Stocker le modèle entraîné
                    self.prediction_model = rf_model
                    
                    # Créer une grille complète de prédictions pour toutes les combinaisons possibles
                    grid_data = []
                    for hour in range(24):
                        for day in range(7):
                            for month in range(1, 13):
                                is_weekend = 1 if day >= 5 else 0
                                is_rush_hour = 1 if (hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 18) else 0
                                is_night = 1 if hour >= 20 or hour <= 5 else 0
                                grid_data.append([hour, day, month, is_weekend, is_rush_hour, is_night])
                    
                    grid_df = pd.DataFrame(grid_data, columns=['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour', 'is_night'])
                    grid_df['predicted_risk'] = rf_model.predict(grid_df)
                    
                    # Normaliser les scores de risque prédits
                    max_risk = grid_df['predicted_risk'].max()
                    if max_risk > 0:
                        grid_df['predicted_risk'] = grid_df['predicted_risk'] / max_risk
                    
                    # Stocker les prédictions dans un dictionnaire
                    self.prediction_grid = {}
                    for _, row in grid_df.iterrows():
                        key = (int(row['hour']), int(row['day_of_week']), int(row['month']))
                        self.prediction_grid[key] = float(row['predicted_risk'])
                except Exception as e:
                    print(f"Erreur lors de l'entraînement du modèle prédictif: {e}")
                    self.prediction_model = None
                    self.prediction_grid = None
            else:
                print("Pas assez de données pour entraîner un modèle prédictif avancé.")
                self.prediction_model = None
                self.prediction_grid = None
        else:
            print("Aucune donnée disponible pour entraîner un modèle prédictif.")
            self.prediction_model = None
            self.prediction_grid = None

    def predict_risk_at_time(self, hour=None, day_of_week=None, month=None, date_time=None):
        """
        Prédit le niveau de risque à un moment spécifique
        
        Args:
            hour (int, optional): Heure (0-23)
            day_of_week (int/str, optional): Jour de la semaine (0-6 ou nom du jour)
            month (int, optional): Mois (1-12)
            date_time (datetime, optional): Date et heure complètes
            
        Returns:
            dict: Dictionnaire contenant les informations de risque
        """
        # Convertir une datetime complète en composants individuels si fournie
        if date_time is not None:
            hour = date_time.hour
            # Convertir le jour de la semaine en indice (0=Monday, 6=Sunday)
            day_of_week = date_time.weekday()
            month = date_time.month
        
        # Convertir le nom du jour en indice si nécessaire
        if isinstance(day_of_week, str):
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if day_of_week in day_names:
                day_of_week = day_names.index(day_of_week)
            else:
                # Essayer une correspondance partielle ou insensible à la casse
                for i, name in enumerate(day_names):
                    if name.lower().startswith(day_of_week.lower()):
                        day_of_week = i
                        break
        
        # Utiliser l'heure courante si non spécifiée
        if hour is None:
            hour = datetime.now().hour
        
        # Utiliser le jour courant si non spécifié
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
            
        # Utiliser le mois courant si non spécifié
        if month is None:
            month = datetime.now().month
        
        # Calculer le risque en utilisant le modèle avancé si disponible
        if self.prediction_model is not None and self.prediction_grid is not None:
            key = (int(hour), int(day_of_week), int(month))
            if key in self.prediction_grid:
                risk_score = self.prediction_grid[key]
            else:
                # Calculer les caractéristiques pour la prédiction
                is_weekend = 1 if day_of_week >= 5 else 0
                is_rush_hour = 1 if (hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 18) else 0
                is_night = 1 if hour >= 20 or hour <= 5 else 0
                
                # Utiliser le modèle pour prédire le risque
                features = np.array([[hour, day_of_week, month, is_weekend, is_rush_hour, is_night]])
                risk_score = float(self.prediction_model.predict(features)[0])
                
                # Normaliser le score
                max_risk = max(list(self.prediction_grid.values()))
                if max_risk > 0:
                    risk_score = risk_score / max_risk
        else:
            # Utiliser une méthode simple basée sur les scores de risque précalculés
            # Obtenir les scores de risque pour l'heure, le jour et le mois
            hour_risk = self.hourly_risk_scores.get(hour, 0) if self.hourly_risk_scores else 0
            
            # Convertir l'indice du jour (0-6) en nom de jour
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_name = day_names[day_of_week] if 0 <= day_of_week < 7 else "Unknown"
            day_risk = self.daily_risk_scores.get(day_name, 0) if self.daily_risk_scores else 0
            
            month_risk = self.monthly_risk_scores.get(month, 0) if self.monthly_risk_scores else 0
            
            # Calculer un score de risque pondéré
            risk_score = (hour_risk * 0.5) + (day_risk * 0.3) + (month_risk * 0.2)
        
        # Déterminer la période de la journée
        time_period = None
        for period, hours in self.time_periods.items():
            if hour in hours:
                time_period = period
                break
        
        # Déterminer le niveau de risque
        if risk_score >= 0.75:
            risk_level = "Très élevé"
        elif risk_score >= 0.5:
            risk_level = "Élevé"
        elif risk_score >= 0.25:
            risk_level = "Modéré"
        else:
            risk_level = "Faible"
        
        # Formatter les informations de risque
        day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        month_names = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                      'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
        
        risk_info = {
            'time': {
                'hour': hour,
                'hour_formatted': f"{hour:02d}:00",
                'day_of_week': day_of_week,
                'day_name': day_names[day_of_week] if 0 <= day_of_week < 7 else "Inconnu",
                'month': month,
                'month_name': month_names[month-1] if 1 <= month <= 12 else "Inconnu",
                'period': time_period
            },
            'risk': {
                'score': round(risk_score, 4),
                'level': risk_level,
                'components': {
                    'hour_risk': round(self.hourly_risk_scores.get(hour, 0), 4) if self.hourly_risk_scores else 0,
                    'day_risk': round(self.daily_risk_scores.get(day_names[day_of_week], 0), 4) if self.daily_risk_scores and 0 <= day_of_week < 7 else 0,
                    'month_risk': round(self.monthly_risk_scores.get(month, 0), 4) if self.monthly_risk_scores else 0
                }
            },
            'recommendations': self.get_time_based_recommendations(hour, day_of_week, month, risk_score)
        }
        
        return risk_info
    
    def get_time_based_recommendations(self, hour, day_of_week, month, risk_score):
        """
        Génère des recommandations basées sur le moment et le niveau de risque
        
        Args:
            hour (int): Heure (0-23)
            day_of_week (int): Jour de la semaine (0-6)
            month (int): Mois (1-12)
            risk_score (float): Score de risque
            
        Returns:
            list: Liste de recommandations
        """
        recommendations = []
        
        # Déterminer la période de la journée
        time_period = None
        for period, hours in self.time_periods.items():
            if hour in hours:
                time_period = period
                break
        
        # Recommandations basées sur la période de la journée
        if time_period == 'nuit':
            recommendations.append("Portez des vêtements clairs ou réfléchissants pour être plus visible.")
            recommendations.append("Privilégiez les rues bien éclairées et les passages piétons signalés.")
            recommendations.append("Soyez particulièrement vigilant aux intersections.")
        elif time_period == 'matin' or time_period == 'soir':
            if 7 <= hour <= 9 or 16 <= hour <= 18:  # Heures de pointe
                recommendations.append("Période de trafic dense, soyez particulièrement attentif aux véhicules tournant aux intersections.")
                recommendations.append("Laissez-vous plus de temps pour traverser.")
                recommendations.append("Privilégiez les passages piétons avec feux de signalisation.")
        
        # Recommandations basées sur le jour de la semaine
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day_of_week] if 0 <= day_of_week < 7 else "Unknown"
        
        if day_name in ['Friday', 'Saturday']:
            recommendations.append("Le weekend, attention accrue à la circulation et aux conducteurs potentiellement distraits.")
            if hour >= 20:
                recommendations.append("Nuit de weekend: risque accru de conducteurs sous influence, redoublez de vigilance.")
        
        # Recommandations basées sur le mois/saison
        if 11 <= month <= 12 or 1 <= month <= 2:  # Hiver
            recommendations.append("Conditions hivernales: attention aux trottoirs glissants et à la visibilité réduite.")
            recommendations.append("Les journées plus courtes réduisent la visibilité, portez des vêtements clairs.")
        elif 6 <= month <= 8:  # Été
            recommendations.append("Période estivale: plus de touristes et de trafic dans certaines zones.")
        
        # Recommandations basées sur le niveau de risque
        if risk_score >= 0.75:  # Risque très élevé
            recommendations.append("⚠️ NIVEAU DE RISQUE TRÈS ÉLEVÉ: redoublez de vigilance.")
            recommendations.append("Évitez de traverser en dehors des passages piétons signalés.")
            recommendations.append("Attendez que les véhicules soient complètement arrêtés avant de traverser.")
        elif risk_score >= 0.5:  # Risque élevé
            recommendations.append("⚠️ NIVEAU DE RISQUE ÉLEVÉ: soyez particulièrement vigilant.")
            recommendations.append("Regardez bien des deux côtés avant de traverser, même aux passages piétons.")
        
        return recommendations
    
    def get_high_risk_times(self, top_n=5):
        """
        Retourne les périodes à plus haut risque
        
        Args:
            top_n (int): Nombre de périodes à retourner
            
        Returns:
            dict: Dictionnaire contenant les périodes à haut risque
        """
        high_risk_times = {
            'hours': [],
            'days': [],
            'months': []
        }
        
        # Trouver les heures à plus haut risque
        if self.hourly_risk_scores:
            # Trier les heures par score de risque décroissant
            sorted_hours = sorted(
                self.hourly_risk_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            # Formater les heures à haut risque
            for hour, score in sorted_hours:
                period = None
                for p, hours in self.time_periods.items():
                    if hour in hours:
                        period = p
                        break
                
                high_risk_times['hours'].append({
                    'hour': hour,
                    'hour_formatted': f"{hour:02d}:00",
                    'risk_score': round(score, 4),
                    'period': period
                })
        
        # Trouver les jours à plus haut risque
        if self.daily_risk_scores:
            # Trier les jours par score de risque décroissant
            sorted_days = sorted(
                self.daily_risk_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            # Formater les jours à haut risque
            day_names_fr = {
                'Monday': 'Lundi',
                'Tuesday': 'Mardi',
                'Wednesday': 'Mercredi',
                'Thursday': 'Jeudi',
                'Friday': 'Vendredi',
                'Saturday': 'Samedi',
                'Sunday': 'Dimanche'
            }
            
            for day, score in sorted_days:
                high_risk_times['days'].append({
                    'day': day,
                    'day_fr': day_names_fr.get(day, day),
                    'risk_score': round(score, 4)
                })
        
        # Trouver les mois à plus haut risque
        if self.monthly_risk_scores:
            # Trier les mois par score de risque décroissant
            sorted_months = sorted(
                self.monthly_risk_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            # Formater les mois à haut risque
            month_names = [
                'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'
            ]
            
            for month, score in sorted_months:
                # Vérifier que le mois est dans la plage valide
                if 1 <= month <= 12:
                    high_risk_times['months'].append({
                        'month': month,
                        'month_name': month_names[month-1],
                        'risk_score': round(score, 4)
                    })
        
        return high_risk_times
    
    def get_risk_heatmap_data(self):
        """
        Génère les données pour une heatmap du risque par heure et jour de la semaine
        
        Returns:
            dict: Données pour la heatmap
        """
        # Initialiser une matrice 7x24 (jours x heures)
        risk_matrix = np.zeros((7, 24))
        
        # Si nous avons un modèle avancé, utiliser les prédictions
        if self.prediction_grid:
            # Remplir la matrice avec les prédictions pour le mois courant
            current_month = datetime.now().month
            
            for day in range(7):
                for hour in range(24):
                    key = (hour, day, current_month)
                    if key in self.prediction_grid:
                        risk_matrix[day, hour] = self.prediction_grid[key]
        else:
            # Sinon, utiliser les scores de risque horaires et journaliers
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Si nous avons des scores horaires et journaliers
            if self.hourly_risk_scores and self.daily_risk_scores:
                for day_idx, day_name in enumerate(day_names):
                    day_risk = self.daily_risk_scores.get(day_name, 0)
                    
                    for hour in range(24):
                        hour_risk = self.hourly_risk_scores.get(hour, 0)
                        # Combiner les scores (moyenne pondérée)
                        risk_matrix[day_idx, hour] = (hour_risk * 0.7) + (day_risk * 0.3)
            # Si nous avons seulement des scores horaires
            elif self.hourly_risk_scores:
                for hour in range(24):
                    hour_risk = self.hourly_risk_scores.get(hour, 0)
                    risk_matrix[:, hour] = hour_risk
            # Si nous avons seulement des scores journaliers
            elif self.daily_risk_scores:
                for day_idx, day_name in enumerate(day_names):
                    day_risk = self.daily_risk_scores.get(day_name, 0)
                    risk_matrix[day_idx, :] = day_risk
        
        # Formater les données pour la sortie
        heatmap_data = {
            'matrix': risk_matrix.tolist(),
            'days': ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'],
            'hours': [f"{hour:02d}:00" for hour in range(24)]
        }
        
        return heatmap_data
    
    def save_model(self):
        """
        Sauvegarde le modèle temporel
        
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            # Créer le répertoire si nécessaire
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Créer un dictionnaire avec toutes les données à sauvegarder
            model_data = {
                'time_risk_data': self.time_risk_data,
                'hourly_risk_scores': self.hourly_risk_scores,
                'daily_risk_scores': self.daily_risk_scores,
                'monthly_risk_scores': self.monthly_risk_scores,
                'prediction_model': self.prediction_model,
                'prediction_grid': self.prediction_grid if hasattr(self, 'prediction_grid') else None,
                'time_periods': self.time_periods
            }
            
            # Sauvegarder le modèle
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Modèle temporel sauvegardé à {self.model_path}")
            return True
        
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle temporel: {e}")
            return False
    
    def load_model(self):
        """
        Charge un modèle temporel sauvegardé
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            # Vérifier si le fichier existe
            if not self.model_path.exists():
                print(f"Aucun modèle temporel trouvé à {self.model_path}")
                return False
            
            # Charger le modèle
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Charger les attributs
            self.time_risk_data = model_data.get('time_risk_data')
            self.hourly_risk_scores = model_data.get('hourly_risk_scores')
            self.daily_risk_scores = model_data.get('daily_risk_scores')
            self.monthly_risk_scores = model_data.get('monthly_risk_scores')
            self.prediction_model = model_data.get('prediction_model')
            self.prediction_grid = model_data.get('prediction_grid')
            self.time_periods = model_data.get('time_periods', self.time_periods)
            
            print(f"Modèle temporel chargé avec succès depuis {self.model_path}")
            return True
        
        except Exception as e:
            print(f"Erreur lors du chargement du modèle temporel: {e}")
            return False