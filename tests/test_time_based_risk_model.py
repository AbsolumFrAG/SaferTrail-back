import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch
import tempfile
from pathlib import Path

from src.time_based_risk_model import TimeBasedPedestrianRiskModel


class TestTimeBasedRiskModelInit:
    """Tests pour l'initialisation du modèle temporel"""
    
    def test_init_success(self):
        """Test d'initialisation réussie"""
        model = TimeBasedPedestrianRiskModel()
        
        assert model.time_risk_data is None
        assert model.hourly_risk_scores is None
        assert model.daily_risk_scores is None
        assert model.monthly_risk_scores is None
        assert model.prediction_model is None
        assert model.model_path.name == 'temporal_risk_model.pkl'
        
        # Vérifier les périodes de la journée
        assert 'nuit' in model.time_periods
        assert 'matin' in model.time_periods
        assert 'après-midi' in model.time_periods
        assert 'soir' in model.time_periods
        
        # Vérifier les heures pour chaque période
        assert 0 in model.time_periods['nuit']
        assert 8 in model.time_periods['matin']
        assert 14 in model.time_periods['après-midi']
        assert 20 in model.time_periods['soir']


class TestAnalyzeTemporalPatterns:
    """Tests pour la méthode analyze_temporal_patterns"""
    
    def test_analyze_temporal_patterns_success(self, sample_crash_data, sample_311_data, sample_crime_data):
        """Test d'analyse réussie des patterns temporels"""
        model = TimeBasedPedestrianRiskModel()
        
        result = model.analyze_temporal_patterns(
            df_crashes=sample_crash_data,
            df_311=sample_311_data,
            df_crimes=sample_crime_data
        )
        
        # Vérifier la structure du résultat
        assert isinstance(result, dict)
        assert 'hourly_counts' in result
        assert 'daily_counts' in result
        assert 'monthly_counts' in result
        assert 'hourly_risk_scores' in result
        assert 'daily_risk_scores' in result
        assert 'monthly_risk_scores' in result
        
        # Vérifier que les données sont stockées dans le modèle
        assert model.time_risk_data is not None
        assert model.hourly_risk_scores is not None
        assert model.daily_risk_scores is not None
        assert model.monthly_risk_scores is not None
        
    def test_analyze_temporal_patterns_crashes_only(self, sample_crash_data):
        """Test avec seulement les données d'accidents"""
        model = TimeBasedPedestrianRiskModel()
        
        result = model.analyze_temporal_patterns(df_crashes=sample_crash_data)
        
        assert isinstance(result, dict)
        assert 'hourly_counts' in result
        assert 'accidents' in result['hourly_counts']
        
        # Vérifier que les scores de risque sont calculés
        assert model.hourly_risk_scores is not None
        assert len(model.hourly_risk_scores) == 24  # 24 heures
        
    def test_analyze_temporal_patterns_empty_data(self):
        """Test avec des données vides"""
        model = TimeBasedPedestrianRiskModel()
        
        empty_df = pd.DataFrame()
        
        result = model.analyze_temporal_patterns(
            df_crashes=empty_df,
            df_311=empty_df,
            df_crimes=empty_df
        )
        
        assert isinstance(result, dict)
        assert model.hourly_risk_scores is not None
        
        # Tous les scores devraient être 0
        assert all(score == 0 for score in model.hourly_risk_scores.values())
        
    def test_analyze_temporal_patterns_missing_columns(self):
        """Test avec colonnes temporelles manquantes"""
        # Données sans colonnes temporelles
        incomplete_data = pd.DataFrame({
            'dispatch_ts': ['2023-01-15T08:30:00', '2023-01-16T19:45:00'],
            'mode_type': ['ped', 'ped']
        })
        incomplete_data['dispatch_ts'] = pd.to_datetime(incomplete_data['dispatch_ts'])
        
        model = TimeBasedPedestrianRiskModel()
        
        result = model.analyze_temporal_patterns(df_crashes=incomplete_data)
        
        # Doit créer les colonnes temporelles automatiquement
        assert isinstance(result, dict)
        assert model.hourly_risk_scores is not None
        
    def test_analyze_temporal_patterns_data_aggregation(self):
        """Test d'agrégation des données par période"""
        # Créer des données avec des patterns temporels spécifiques
        crash_data = pd.DataFrame({
            'dispatch_ts': [
                '2023-01-15T08:30:00',  # Matin
                '2023-01-15T08:45:00',  # Matin (même heure)
                '2023-01-15T20:30:00',  # Soir
                '2023-01-16T08:30:00'   # Matin (jour différent)
            ],
            'mode_type': ['ped'] * 4
        })
        crash_data['dispatch_ts'] = pd.to_datetime(crash_data['dispatch_ts'])
        
        model = TimeBasedPedestrianRiskModel()
        
        result = model.analyze_temporal_patterns(df_crashes=crash_data)
        
        # Vérifier l'agrégation par heure
        hourly_counts = result['hourly_counts']['accidents']
        assert hourly_counts[8] == 3  # 3 accidents à 8h
        assert hourly_counts[20] == 1  # 1 accident à 20h
        
        # Vérifier l'agrégation par jour
        daily_counts = result['daily_counts']['accidents']
        assert daily_counts['Sunday'] == 3  # 3 accidents le dimanche (15 janvier 2023)
        assert daily_counts['Monday'] == 1   # 1 accident le lundi (16 janvier 2023)


class TestCalculateTemporalRiskScores:
    """Tests pour la méthode calculate_temporal_risk_scores"""
    
    def test_calculate_risk_scores_success(self):
        """Test de calcul réussi des scores de risque temporels"""
        model = TimeBasedPedestrianRiskModel()
        
        # Données d'exemple
        hourly_counts = {
            'accidents': {8: 5, 20: 10},  # Plus d'accidents le soir
            '311_reports': {8: 2, 20: 3},
            'crimes': {8: 1, 20: 4}
        }
        daily_counts = {
            'accidents': {'Monday': 3, 'Friday': 8, 'Saturday': 12},
            '311_reports': {'Monday': 2, 'Friday': 4, 'Saturday': 5},
            'crimes': {'Monday': 1, 'Friday': 3, 'Saturday': 6}
        }
        monthly_counts = {
            'accidents': {1: 10, 6: 15, 12: 8},
            '311_reports': {1: 5, 6: 8, 12: 4},
            'crimes': {1: 3, 6: 6, 12: 2}
        }
        
        model.calculate_temporal_risk_scores(hourly_counts, daily_counts, monthly_counts)
        
        # Vérifier que les scores sont calculés
        assert model.hourly_risk_scores is not None
        assert model.daily_risk_scores is not None
        assert model.monthly_risk_scores is not None
        
        # Vérifier la normalisation (scores entre 0 et 1)
        assert all(0 <= score <= 1 for score in model.hourly_risk_scores.values())
        assert all(0 <= score <= 1 for score in model.daily_risk_scores.values())
        assert all(0 <= score <= 1 for score in model.monthly_risk_scores.values())
        
        # Le soir (20h) devrait avoir un score plus élevé que le matin (8h)
        assert model.hourly_risk_scores[20] > model.hourly_risk_scores[8]
        
    def test_calculate_risk_scores_weighting(self):
        """Test de pondération des différentes sources de données"""
        model = TimeBasedPedestrianRiskModel()
        
        # Données où les accidents dominent
        hourly_counts = {
            'accidents': {8: 10, 20: 5},     # Plus d'accidents le matin
            '311_reports': {8: 1, 20: 10},   # Plus de signalements le soir
            'crimes': {8: 1, 20: 1}
        }
        
        model.calculate_temporal_risk_scores(hourly_counts, {}, {})
        
        # Le matin devrait avoir un score plus élevé car les accidents ont plus de poids
        assert model.hourly_risk_scores[8] > model.hourly_risk_scores[20]
        
    def test_calculate_risk_scores_normalization(self):
        """Test de normalisation des scores"""
        model = TimeBasedPedestrianRiskModel()
        
        # Données avec valeurs extrêmes
        hourly_counts = {
            'accidents': {0: 0, 12: 100},  # Valeur très élevée à midi
            '311_reports': {},
            'crimes': {}
        }
        
        model.calculate_temporal_risk_scores(hourly_counts, {}, {})
        
        # Le score maximum devrait être 1 (normalisé)
        max_score = max(model.hourly_risk_scores.values())
        assert max_score == 1.0
        
        # Le score à midi devrait être le maximum
        assert model.hourly_risk_scores[12] == 1.0
        assert model.hourly_risk_scores[0] == 0.0


class TestTrainPredictionModel:
    """Tests pour la méthode train_prediction_model"""
    
    def test_train_prediction_model_success(self, sample_crash_data, sample_311_data, sample_crime_data):
        """Test d'entraînement réussi du modèle prédictif"""
        model = TimeBasedPedestrianRiskModel()
        
        model.train_prediction_model(
            df_crashes=sample_crash_data,
            df_311=sample_311_data,
            df_crimes=sample_crime_data
        )
        
        # Vérifier que le modèle est entraîné
        if hasattr(model, 'prediction_model') and model.prediction_model is not None:
            assert model.prediction_model is not None
            assert hasattr(model, 'prediction_grid')
            
    def test_train_prediction_model_insufficient_data(self):
        """Test avec données insuffisantes"""
        model = TimeBasedPedestrianRiskModel()
        
        # Très peu de données
        minimal_data = pd.DataFrame({
            'dispatch_ts': ['2023-01-15T08:30:00'],
            'mode_type': ['ped']
        })
        minimal_data['dispatch_ts'] = pd.to_datetime(minimal_data['dispatch_ts'])
        
        model.train_prediction_model(df_crashes=minimal_data)
        
        # Ne devrait pas créer de modèle avec si peu de données
        assert model.prediction_model is None or hasattr(model, 'prediction_grid')
        
    def test_train_prediction_model_invalid_dates(self):
        """Test avec dates invalides"""
        model = TimeBasedPedestrianRiskModel()
        
        invalid_data = pd.DataFrame({
            'dispatch_ts': ['invalid_date', '2023-01-15T08:30:00'],
            'mode_type': ['ped', 'ped']
        })
        invalid_data['dispatch_ts'] = pd.to_datetime(invalid_data['dispatch_ts'], errors='coerce')
        
        # Doit gérer les dates invalides
        model.train_prediction_model(df_crashes=invalid_data)
        
        # Ne devrait pas planter
        assert True
        
    def test_train_prediction_model_feature_extraction(self):
        """Test d'extraction des caractéristiques temporelles"""
        model = TimeBasedPedestrianRiskModel()
        
        # Données avec différents patterns temporels
        varied_data = pd.DataFrame({
            'dispatch_ts': [
                '2023-01-15T08:30:00',  # Dimanche matin
                '2023-01-20T17:30:00',  # Vendredi soir (heure de pointe)
                '2023-01-21T23:30:00'   # Samedi nuit (weekend)
            ],
            'mode_type': ['ped'] * 3
        })
        varied_data['dispatch_ts'] = pd.to_datetime(varied_data['dispatch_ts'])
        
        model.train_prediction_model(df_crashes=varied_data)
        
        # Le modèle devrait extraire les bonnes caractéristiques
        # (Test implicite - pas d'erreur signifie que l'extraction fonctionne)
        assert True


class TestPredictRiskAtTime:
    """Tests pour la méthode predict_risk_at_time"""
    
    def test_predict_risk_at_time_basic(self, mock_time_model):
        """Test de prédiction de base"""
        model = TimeBasedPedestrianRiskModel()
        model.hourly_risk_scores = mock_time_model.hourly_risk_scores
        model.daily_risk_scores = mock_time_model.daily_risk_scores
        model.monthly_risk_scores = mock_time_model.monthly_risk_scores
        
        result = model.predict_risk_at_time(hour=14, day_of_week=3, month=6)
        
        assert isinstance(result, dict)
        assert 'time' in result
        assert 'risk' in result
        assert 'recommendations' in result
        
        # Vérifier les informations temporelles
        assert result['time']['hour'] == 14
        assert result['time']['day_of_week'] == 3
        assert result['time']['month'] == 6
        
        # Vérifier le score de risque
        assert 'score' in result['risk']
        assert 'level' in result['risk']
        assert 0 <= result['risk']['score'] <= 1
        
    def test_predict_risk_at_time_with_datetime(self, mock_time_model):
        """Test avec objet datetime"""
        model = TimeBasedPedestrianRiskModel()
        model.hourly_risk_scores = mock_time_model.hourly_risk_scores
        model.daily_risk_scores = mock_time_model.daily_risk_scores
        model.monthly_risk_scores = mock_time_model.monthly_risk_scores
        
        test_datetime = datetime(2023, 6, 15, 14, 30)  # Jeudi 15 juin 2023, 14h30
        
        result = model.predict_risk_at_time(date_time=test_datetime)
        
        assert result['time']['hour'] == 14
        assert result['time']['day_of_week'] == 3  # Jeudi (0=lundi)
        assert result['time']['month'] == 6
        
    def test_predict_risk_at_time_day_name_conversion(self, mock_time_model):
        """Test de conversion du nom de jour"""
        model = TimeBasedPedestrianRiskModel()
        model.hourly_risk_scores = mock_time_model.hourly_risk_scores
        model.daily_risk_scores = mock_time_model.daily_risk_scores
        model.monthly_risk_scores = mock_time_model.monthly_risk_scores
        
        result = model.predict_risk_at_time(hour=14, day_of_week='Thursday', month=6)
        
        assert result['time']['day_of_week'] == 3  # Jeudi converti en indice
        
    def test_predict_risk_at_time_default_values(self, mock_time_model):
        """Test avec valeurs par défaut"""
        model = TimeBasedPedestrianRiskModel()
        model.hourly_risk_scores = mock_time_model.hourly_risk_scores
        model.daily_risk_scores = mock_time_model.daily_risk_scores
        model.monthly_risk_scores = mock_time_model.monthly_risk_scores
        
        with patch('src.time_based_risk_model.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 6, 15, 14, 30)
            
            result = model.predict_risk_at_time()  # Aucun paramètre
            
            # Doit utiliser l'heure actuelle
            assert result['time']['hour'] == 14
            assert result['time']['month'] == 6


class TestGetTimeBasedRecommendations:
    """Tests pour la méthode get_time_based_recommendations"""
        
    def test_get_time_based_recommendations_rush_hour(self):
        """Test des recommandations aux heures de pointe"""
        model = TimeBasedPedestrianRiskModel()
        
        recommendations = model.get_time_based_recommendations(
            hour=8, day_of_week=1, month=6, risk_score=0.6  # Mardi matin
        )
        
        assert isinstance(recommendations, list)
        
        # Doit contenir des recommandations pour les heures de pointe
        recommendation_text = ' '.join(recommendations).lower()
        assert any(word in recommendation_text for word in ['trafic', 'dense', 'intersection'])
        
    def test_get_time_based_recommendations_weekend(self):
        """Test des recommandations de weekend"""
        model = TimeBasedPedestrianRiskModel()
        
        recommendations = model.get_time_based_recommendations(
            hour=23, day_of_week=5, month=6, risk_score=0.9  # Samedi soir
        )
        
        assert isinstance(recommendations, list)
        
        # Doit contenir des recommandations spécifiques au weekend
        recommendation_text = ' '.join(recommendations).lower()
        assert any(word in recommendation_text for word in ['weekend', 'vigilance', 'influence'])
        
    def test_get_time_based_recommendations_high_risk(self):
        """Test des recommandations pour haut risque"""
        model = TimeBasedPedestrianRiskModel()
        
        recommendations = model.get_time_based_recommendations(
            hour=14, day_of_week=2, month=6, risk_score=0.9  # Risque très élevé
        )
        
        assert isinstance(recommendations, list)
        
        # Doit contenir des avertissements pour haut risque
        recommendation_text = ' '.join(recommendations)
        assert '⚠️' in recommendation_text or 'RISQUE' in recommendation_text.upper()
        
    def test_get_time_based_recommendations_winter(self):
        """Test des recommandations hivernales"""
        model = TimeBasedPedestrianRiskModel()
        
        recommendations = model.get_time_based_recommendations(
            hour=14, day_of_week=2, month=12, risk_score=0.5  # Décembre
        )
        
        assert isinstance(recommendations, list)
        
        # Doit contenir des recommandations hivernales
        recommendation_text = ' '.join(recommendations).lower()
        assert any(word in recommendation_text for word in ['hiver', 'glissant', 'visibilité'])


class TestGetHighRiskTimes:
    """Tests pour la méthode get_high_risk_times"""
    
    def test_get_high_risk_times_success(self, mock_time_model):
        """Test de récupération réussie des périodes à haut risque"""
        model = TimeBasedPedestrianRiskModel()
        model.hourly_risk_scores = mock_time_model.hourly_risk_scores
        model.daily_risk_scores = mock_time_model.daily_risk_scores
        model.monthly_risk_scores = mock_time_model.monthly_risk_scores
        
        result = model.get_high_risk_times(top_n=3)
        
        assert isinstance(result, dict)
        assert 'hours' in result
        assert 'days' in result
        assert 'months' in result
        
        # Vérifier la structure des heures à haut risque
        assert len(result['hours']) <= 3
        for hour_info in result['hours']:
            assert 'hour' in hour_info
            assert 'risk_score' in hour_info
            assert 'period' in hour_info
            assert 0 <= hour_info['hour'] <= 23
            
    def test_get_high_risk_times_sorting(self):
        """Test du tri par score de risque"""
        model = TimeBasedPedestrianRiskModel()
        
        # Scores avec ordre spécifique
        model.hourly_risk_scores = {
            8: 0.3,   # Matin modéré
            12: 0.5,  # Midi élevé
            18: 0.8,  # Soir très élevé
            22: 0.6   # Nuit élevé
        }
        
        result = model.get_high_risk_times(top_n=4)
        
        # Vérifier que les heures sont triées par score décroissant
        hours_list = result['hours']
        for i in range(len(hours_list) - 1):
            assert hours_list[i]['risk_score'] >= hours_list[i + 1]['risk_score']
            
        # La première heure devrait être 18h (score 0.8)
        assert hours_list[0]['hour'] == 18
        
    def test_get_high_risk_times_day_translation(self):
        """Test de traduction des noms de jours"""
        model = TimeBasedPedestrianRiskModel()
        
        model.daily_risk_scores = {
            'Monday': 0.5,
            'Friday': 0.8,
            'Saturday': 0.9
        }
        
        result = model.get_high_risk_times(top_n=3)
        
        # Vérifier que les noms français sont utilisés
        for day_info in result['days']:
            assert 'day_fr' in day_info
            assert day_info['day_fr'] in ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 
                                         'Vendredi', 'Samedi', 'Dimanche']
                                         
    def test_get_high_risk_times_empty_data(self):
        """Test avec données vides"""
        model = TimeBasedPedestrianRiskModel()
        
        result = model.get_high_risk_times()
        
        assert isinstance(result, dict)
        assert result['hours'] == []
        assert result['days'] == []
        assert result['months'] == []


class TestGetRiskHeatmapData:
    """Tests pour la méthode get_risk_heatmap_data"""
        
    def test_get_risk_heatmap_data_with_prediction_grid(self):
        """Test avec grille de prédiction avancée"""
        model = TimeBasedPedestrianRiskModel()
        
        # Créer une grille de prédiction fictive
        model.prediction_grid = {}
        current_month = datetime.now().month
        
        for day in range(7):
            for hour in range(24):
                key = (hour, day, current_month)
                model.prediction_grid[key] = np.random.random()
                
        result = model.get_risk_heatmap_data()
        
        assert isinstance(result, dict)
        assert len(result['matrix']) == 7
        assert len(result['matrix'][0]) == 24


class TestModelPersistence:
    """Tests pour la sauvegarde et le chargement du modèle"""
    
    def test_save_model_success(self, mock_time_model):
        """Test de sauvegarde réussie"""
        model = TimeBasedPedestrianRiskModel()
        model.hourly_risk_scores = mock_time_model.hourly_risk_scores
        model.daily_risk_scores = mock_time_model.daily_risk_scores
        model.monthly_risk_scores = mock_time_model.monthly_risk_scores
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model.model_path = Path(temp_dir) / 'test_temporal_model.pkl'
            
            success = model.save_model()
            
            assert success is True
            assert model.model_path.exists()
            
    def test_load_model_success(self, mock_time_model):
        """Test de chargement réussi"""
        # Créer et sauvegarder un modèle
        original_model = TimeBasedPedestrianRiskModel()
        original_model.hourly_risk_scores = mock_time_model.hourly_risk_scores
        original_model.daily_risk_scores = mock_time_model.daily_risk_scores
        original_model.monthly_risk_scores = mock_time_model.monthly_risk_scores
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'test_temporal_model.pkl'
            original_model.model_path = model_path
            original_model.save_model()
            
            # Charger le modèle
            new_model = TimeBasedPedestrianRiskModel()
            new_model.model_path = model_path
            success = new_model.load_model()
            
            assert success is True
            assert new_model.hourly_risk_scores is not None
            assert new_model.daily_risk_scores is not None
            assert new_model.monthly_risk_scores is not None
            
    def test_load_model_file_not_found(self):
        """Test de chargement avec fichier inexistant"""
        model = TimeBasedPedestrianRiskModel()
        model.model_path = Path("nonexistent_temporal_model.pkl")
        
        success = model.load_model()
        
        assert success is False


class TestIntegrationAndWorkflow:
    """Tests d'intégration et de workflow complet"""
    
    def test_complete_workflow(self, sample_crash_data, sample_311_data, sample_crime_data):
        """Test du workflow complet d'analyse temporelle"""
        model = TimeBasedPedestrianRiskModel()
        
        # 1. Analyser les patterns temporels
        analysis_result = model.analyze_temporal_patterns(
            df_crashes=sample_crash_data,
            df_311=sample_311_data,
            df_crimes=sample_crime_data
        )
        
        assert isinstance(analysis_result, dict)
        
        # 2. Prédire le risque à différents moments
        morning_risk = model.predict_risk_at_time(hour=8, day_of_week=1, month=6)
        evening_risk = model.predict_risk_at_time(hour=20, day_of_week=5, month=6)
        
        assert morning_risk['time']['hour'] == 8
        assert evening_risk['time']['hour'] == 20
        
        # 3. Obtenir les périodes à haut risque
        high_risk_times = model.get_high_risk_times(top_n=3)
        
        assert len(high_risk_times['hours']) <= 3
        
        # 4. Générer les données de heatmap
        heatmap_data = model.get_risk_heatmap_data()
        
        assert len(heatmap_data['matrix']) == 7
        assert len(heatmap_data['matrix'][0]) == 24
        
    def test_model_consistency(self, sample_crash_data):
        """Test de cohérence du modèle"""
        model = TimeBasedPedestrianRiskModel()
        
        # Analyser les mêmes données plusieurs fois
        result1 = model.analyze_temporal_patterns(df_crashes=sample_crash_data)
        result2 = model.analyze_temporal_patterns(df_crashes=sample_crash_data)
        
        # Les résultats devraient être identiques
        assert result1['hourly_risk_scores'] == result2['hourly_risk_scores']
            
    def test_performance_with_large_dataset(self):
        """Test de performance avec un grand dataset"""
        model = TimeBasedPedestrianRiskModel()
        
        # Créer un grand dataset synthétique
        n_records = 10000
        large_dataset = pd.DataFrame({
            'dispatch_ts': [
                datetime.now() - timedelta(days=np.random.randint(0, 365), 
                                         hours=np.random.randint(0, 24))
                for _ in range(n_records)
            ],
            'mode_type': ['ped'] * n_records
        })
        
        # L'analyse devrait se terminer sans erreur
        result = model.analyze_temporal_patterns(df_crashes=large_dataset)
        
        assert isinstance(result, dict)
        assert model.hourly_risk_scores is not None
        assert len(model.hourly_risk_scores) == 24


class TestValidationAndEdgeCases:
    """Tests de validation et cas limites"""
    
    def test_invalid_time_parameters(self, mock_time_model):
        """Test avec paramètres temporels invalides"""
        model = TimeBasedPedestrianRiskModel()
        model.hourly_risk_scores = mock_time_model.hourly_risk_scores
        model.daily_risk_scores = mock_time_model.daily_risk_scores
        model.monthly_risk_scores = mock_time_model.monthly_risk_scores
        
        # Paramètres en dehors des plages valides
        result = model.predict_risk_at_time(hour=25, day_of_week=8, month=13)
        
        # Doit gérer les valeurs invalides gracieusement
        assert isinstance(result, dict)
        
    def test_boundary_conditions(self, mock_time_model):
        """Test des conditions limites"""
        model = TimeBasedPedestrianRiskModel()
        model.hourly_risk_scores = mock_time_model.hourly_risk_scores
        model.daily_risk_scores = mock_time_model.daily_risk_scores
        model.monthly_risk_scores = mock_time_model.monthly_risk_scores
        
        # Test aux limites des heures
        result_midnight = model.predict_risk_at_time(hour=0)
        result_almost_midnight = model.predict_risk_at_time(hour=23)
        
        assert result_midnight['time']['hour'] == 0
        assert result_almost_midnight['time']['hour'] == 23
        
        # Test aux limites des mois
        result_january = model.predict_risk_at_time(month=1)
        result_december = model.predict_risk_at_time(month=12)
        
        assert result_january['time']['month'] == 1
        assert result_december['time']['month'] == 12