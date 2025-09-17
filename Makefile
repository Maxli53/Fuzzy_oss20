# Fuzzy OSS20 Trading Platform Makefile

.PHONY: help build up down restart logs clean test install dev prod

help:
	@echo "Fuzzy OSS20 Trading Platform - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make install    - Install all dependencies"
	@echo "  make dev        - Start development environment"
	@echo "  make test       - Run all tests"
	@echo ""
	@echo "Docker:"
	@echo "  make build      - Build Docker containers"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View container logs"
	@echo "  make clean      - Clean up containers and volumes"
	@echo ""
	@echo "Production:"
	@echo "  make prod       - Start production environment"

# Install dependencies
install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Dependencies installed successfully!"

# Development environment
dev:
	@echo "Starting development environment..."
	docker-compose up postgres redis -d
	@echo "Starting backend..."
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "Starting frontend..."
	cd frontend && npm start &
	@echo "Development environment started!"
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"

# Docker commands
build:
	docker-compose build --no-cache

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -f

# Testing
test:
	@echo "Running backend tests..."
	cd backend && pytest
	@echo "Running frontend tests..."
	cd frontend && npm test
	@echo "Running E2E tests..."
	cd tests && npx playwright test

# Production
prod:
	docker-compose --profile production up -d
	@echo "Production environment started!"
	@echo "Application: http://localhost"
	@echo "Grafana: http://localhost:3001 (admin/admin123)"

# Database management
db-migrate:
	cd backend && alembic upgrade head

db-rollback:
	cd backend && alembic downgrade -1

db-reset:
	cd backend && alembic downgrade base && alembic upgrade head

# IQFeed connection check
check-iqfeed:
	@echo "Checking IQFeed connection..."
	python -c "import sys; sys.path.append('.'); from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector; c = IQFeedCollector(); print('Connected!' if c.ensure_connection() else 'Failed to connect')"

# ArcticDB check
check-arctic:
	@echo "Checking ArcticDB..."
	python -c "from arcticdb import Arctic; a = Arctic('lmdb://arctic_storage'); print(f'Libraries: {list(a.list_libraries())}')"

# Start specific services
backend-only:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend-only:
	cd frontend && npm start

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3001"
	open http://localhost:9090
	open http://localhost:3001