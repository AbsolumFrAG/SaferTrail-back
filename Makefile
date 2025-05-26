.PHONY: help test test-unit test-integration test-api test-slow test-quick test-coverage test-parallel test-watch clean-test setup-test lint

# Variables
PYTHON := python3
PYTEST := PYTHONPATH=. pytest
PIP := pip
TEST_DIR := tests
SRC_DIR := src
COVERAGE_DIR := htmlcov
REPORTS_DIR := test-reports

# Couleurs pour l'affichage
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
WHITE := \033[37m
RESET := \033[0m

# Aide par défaut
help: ## Affiche cette aide
	@echo "$(CYAN)SaferTrail - Commandes de test disponibles:$(RESET)"
	@echo ""
	@egrep '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Exemples d'utilisation:$(RESET)"
	@echo "  make test-quick          # Tests rapides uniquement"
	@echo "  make test-coverage       # Tests avec rapport de couverture"
	@echo "  make test-api            # Tests de l'API seulement"
	@echo "  make test-parallel       # Tests en parallèle"

# Installation des dépendances de test
setup-test: ## Installe les dépendances de test
	@echo "$(BLUE)Installation des dépendances de test...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov pytest-xdist pytest-mock pytest-timeout pytest-sugar pytest-html pytest-json-report
	@echo "$(GREEN)✓ Dépendances installées$(RESET)"

# Tests principaux
test: ## Lance tous les tests
	@echo "$(BLUE)Lancement de tous les tests...$(RESET)"
	$(PYTEST) $(TEST_DIR) --html=$(REPORTS_DIR)/report.html --self-contained-html
	@echo "$(GREEN)✓ Tests terminés$(RESET)"

test-unit: ## Lance uniquement les tests unitaires
	@echo "$(BLUE)Lancement des tests unitaires...$(RESET)"
	$(PYTEST) -m "unit" $(TEST_DIR) -v
	@echo "$(GREEN)✓ Tests unitaires terminés$(RESET)"

test-integration: ## Lance uniquement les tests d'intégration
	@echo "$(BLUE)Lancement des tests d'intégration...$(RESET)"
	$(PYTEST) -m "integration" $(TEST_DIR) -v --timeout=60
	@echo "$(GREEN)✓ Tests d'intégration terminés$(RESET)"

test-api: ## Lance uniquement les tests de l'API
	@echo "$(BLUE)Lancement des tests API...$(RESET)"
	$(PYTEST) -m "api" $(TEST_DIR)/test_api.py -v
	@echo "$(GREEN)✓ Tests API terminés$(RESET)"

test-routing: ## Lance uniquement les tests de routage
	@echo "$(BLUE)Lancement des tests de routage...$(RESET)"
	$(PYTEST) -m "routing" $(TEST_DIR)/test_alt_routing.py -v
	@echo "$(GREEN)✓ Tests de routage terminés$(RESET)"

test-temporal: ## Lance uniquement les tests temporels
	@echo "$(BLUE)Lancement des tests temporels...$(RESET)"
	$(PYTEST) -m "temporal" $(TEST_DIR)/test_time_based_risk_model.py -v
	@echo "$(GREEN)✓ Tests temporels terminés$(RESET)"

test-geospatial: ## Lance uniquement les tests géospatiaux
	@echo "$(BLUE)Lancement des tests géospatiaux...$(RESET)"
	$(PYTEST) -m "geospatial" $(TEST_DIR)/test_geo_utils.py $(TEST_DIR)/test_data_processor.py -v
	@echo "$(GREEN)✓ Tests géospatiaux terminés$(RESET)"

test-data: ## Lance uniquement les tests de données
	@echo "$(BLUE)Lancement des tests de données...$(RESET)"
	$(PYTEST) -m "data" $(TEST_DIR)/test_data_loader.py $(TEST_DIR)/test_data_processor.py -v
	@echo "$(GREEN)✓ Tests de données terminés$(RESET)"

test-model: ## Lance uniquement les tests des modèles
	@echo "$(BLUE)Lancement des tests des modèles...$(RESET)"
	$(PYTEST) -m "model" $(TEST_DIR)/test_risk_model.py $(TEST_DIR)/test_time_based_risk_model.py -v
	@echo "$(GREEN)✓ Tests des modèles terminés$(RESET)"

# Tests rapides et lents
test-quick: ## Lance uniquement les tests rapides (< 5 secondes)
	@echo "$(BLUE)Lancement des tests rapides...$(RESET)"
	$(PYTEST) -m "not slow" $(TEST_DIR) -x --timeout=5
	@echo "$(GREEN)✓ Tests rapides terminés$(RESET)"

test-slow: ## Lance uniquement les tests lents
	@echo "$(BLUE)Lancement des tests lents...$(RESET)"
	$(PYTEST) -m "slow" $(TEST_DIR) -v --timeout=300
	@echo "$(GREEN)✓ Tests lents terminés$(RESET)"

# Tests avec couverture
test-coverage: ## Lance les tests avec rapport de couverture
	@echo "$(BLUE)Lancement des tests avec couverture...$(RESET)"
	$(PYTEST) $(TEST_DIR) \
		--cov=$(SRC_DIR) \
		--cov-report=html:$(COVERAGE_DIR) \
		--cov-report=term-missing \
		--cov-report=xml:coverage.xml \
		--cov-fail-under=80
	@echo "$(GREEN)✓ Tests avec couverture terminés$(RESET)"
	@echo "$(CYAN)📊 Rapport de couverture disponible dans $(COVERAGE_DIR)/index.html$(RESET)"

test-coverage-report: ## Ouvre le rapport de couverture dans le navigateur
	@if [ -f "$(COVERAGE_DIR)/index.html" ]; then \
		echo "$(BLUE)Ouverture du rapport de couverture...$(RESET)"; \
		python -m webbrowser $(COVERAGE_DIR)/index.html; \
	else \
		echo "$(RED)❌ Rapport de couverture non trouvé. Lancez 'make test-coverage' d'abord.$(RESET)"; \
	fi

# Tests en parallèle
test-parallel: ## Lance les tests en parallèle (plus rapide)
	@echo "$(BLUE)Lancement des tests en parallèle...$(RESET)"
	$(PYTEST) $(TEST_DIR) -n auto --dist=loadfile
	@echo "$(GREEN)✓ Tests parallèles terminés$(RESET)"

# Tests avec surveillance des fichiers
test-watch: ## Lance les tests en mode surveillance (relance automatiquement)
	@echo "$(BLUE)Mode surveillance activé - Les tests se relanceront automatiquement...$(RESET)"
	@echo "$(YELLOW)Appuyez sur Ctrl+C pour arrêter$(RESET)"
	while true; do \
		$(PYTEST) $(TEST_DIR) -x --tb=short || true; \
		echo "$(CYAN)En attente de modifications...$(RESET)"; \
		find $(SRC_DIR) $(TEST_DIR) -name "*.py" | entr -d echo "Fichiers modifiés détectés"; \
	done

# Tests de performance
test-performance: ## Lance les tests de performance
	@echo "$(BLUE)Lancement des tests de performance...$(RESET)"
	$(PYTEST) -m "performance" $(TEST_DIR) -v --timeout=120
	@echo "$(GREEN)✓ Tests de performance terminés$(RESET)"

# Tests avec profiling
test-profile: ## Lance les tests avec profiling
	@echo "$(BLUE)Lancement des tests avec profiling...$(RESET)"
	$(PYTEST) $(TEST_DIR) --profile --profile-svg
	@echo "$(GREEN)✓ Tests avec profiling terminés$(RESET)"

# Tests spécifiques par fichier
test-geo: ## Tests des utilitaires géographiques
	$(PYTEST) $(TEST_DIR)/test_geo_utils.py -v

test-loader: ## Tests du chargeur de données
	$(PYTEST) $(TEST_DIR)/test_data_loader.py -v

test-processor: ## Tests du processeur de données
	$(PYTEST) $(TEST_DIR)/test_data_processor.py -v

test-risk: ## Tests du modèle de risque
	$(PYTEST) $(TEST_DIR)/test_risk_model.py -v

test-time: ## Tests du modèle temporel
	$(PYTEST) $(TEST_DIR)/test_time_based_risk_model.py -v

test-alt: ## Tests du routage ALT
	$(PYTEST) $(TEST_DIR)/test_alt_routing.py -v

# Tests avec options spéciales
test-verbose: ## Lance les tests en mode très verbeux
	@echo "$(BLUE)Lancement des tests en mode verbeux...$(RESET)"
	$(PYTEST) $(TEST_DIR) -vvv --tb=long --showlocals

test-debug: ## Lance les tests en mode debug (s'arrête au premier échec)
	@echo "$(BLUE)Lancement des tests en mode debug...$(RESET)"
	$(PYTEST) $(TEST_DIR) -x -vvv --tb=long --showlocals --pdb

test-failed: ## Relance uniquement les tests qui ont échoué
	@echo "$(BLUE)Relancement des tests échoués...$(RESET)"
	$(PYTEST) --lf $(TEST_DIR) -v

test-new: ## Lance uniquement les nouveaux tests (modifiés)
	@echo "$(BLUE)Lancement des nouveaux tests...$(RESET)"
	$(PYTEST) --nf $(TEST_DIR) -v

# Génération de rapports
test-report: ## Génère un rapport complet des tests
	@echo "$(BLUE)Génération du rapport complet...$(RESET)"
	mkdir -p $(REPORTS_DIR)
	$(PYTEST) $(TEST_DIR) \
		--html=$(REPORTS_DIR)/report.html \
		--self-contained-html \
		--json-report --json-report-file=$(REPORTS_DIR)/report.json \
		--cov=$(SRC_DIR) \
		--cov-report=html:$(COVERAGE_DIR) \
		--junit-xml=$(REPORTS_DIR)/junit.xml
	@echo "$(GREEN)✓ Rapport généré dans $(REPORTS_DIR)/$(RESET)"

# Nettoyage
clean-test: ## Nettoie les fichiers de test temporaires
	@echo "$(BLUE)Nettoyage des fichiers de test...$(RESET)"
	rm -rf $(COVERAGE_DIR)
	rm -rf $(REPORTS_DIR)
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -f coverage.xml
	rm -f junit.xml
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Nettoyage terminé$(RESET)"

# Vérification de l'environnement
check-env: ## Vérifie l'environnement de test
	@echo "$(BLUE)Vérification de l'environnement...$(RESET)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pytest: $(shell PYTHONPATH=. pytest --version)"
	@echo "Répertoire des tests: $(TEST_DIR)"
	@echo "Répertoire source: $(SRC_DIR)"
	@if [ -d "$(TEST_DIR)" ]; then \
		echo "$(GREEN)✓ Répertoire de tests trouvé$(RESET)"; \
	else \
		echo "$(RED)❌ Répertoire de tests non trouvé$(RESET)"; \
	fi
	@if [ -d "$(SRC_DIR)" ]; then \
		echo "$(GREEN)✓ Répertoire source trouvé$(RESET)"; \
	else \
		echo "$(RED)❌ Répertoire source non trouvé$(RESET)"; \
	fi

# Linting des tests
lint-tests: ## Vérifie la qualité du code des tests
	@echo "$(BLUE)Vérification de la qualité des tests...$(RESET)"
	flake8 $(TEST_DIR) --max-line-length=120 --extend-ignore=E203,W503
	black --check $(TEST_DIR)
	isort --check-only $(TEST_DIR)
	@echo "$(GREEN)✓ Qualité des tests vérifiée$(RESET)"

fix-tests: ## Corrige automatiquement les problèmes de style des tests
	@echo "$(BLUE)Correction automatique du style...$(RESET)"
	black $(TEST_DIR)
	isort $(TEST_DIR)
	@echo "$(GREEN)✓ Style corrigé$(RESET)"

# Statistiques des tests
test-stats: ## Affiche les statistiques des tests
	@echo "$(CYAN)📊 Statistiques des tests:$(RESET)"
	@echo "Nombre total de fichiers de tests: $(shell find $(TEST_DIR) -name "test_*.py" | wc -l)"
	@echo "Nombre total de tests: $(shell PYTHONPATH=. $(PYTEST) --collect-only -q $(TEST_DIR) 2>/dev/null | grep "test" | wc -l)"
	@echo "Lignes de code de tests: $(shell find $(TEST_DIR) -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "Taille des tests: $(shell du -sh $(TEST_DIR) | cut -f1)"

# CI/CD
test-ci: ## Lance les tests comme en CI/CD
	@echo "$(BLUE)Lancement des tests CI/CD...$(RESET)"
	$(PYTEST) $(TEST_DIR) \
		--maxfail=1 \
		--disable-warnings \
		--tb=short \
		--cov=$(SRC_DIR) \
		--cov-report=xml \
		--junit-xml=junit.xml
	@echo "$(GREEN)✓ Tests CI/CD terminés$(RESET)"

# Installation complète pour le développement
dev-setup: setup-test ## Installation complète pour le développement
	@echo "$(BLUE)Installation de l'environnement de développement...$(RESET)"
	$(PIP) install black isort flake8 mypy pre-commit
	pre-commit install
	@echo "$(GREEN)✓ Environnement de développement prêt$(RESET)"

# Commande par défaut
.DEFAULT_GOAL := help