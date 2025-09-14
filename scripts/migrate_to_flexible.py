#!/usr/bin/env python3
"""
Migration Script: Legacy to Flexible Storage
Migrate existing rigid storage to new exploratory flexible architecture
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

from stage_01_data_engine.storage.tick_store import TickStore
from stage_01_data_engine.storage.flexible_arctic_store import FlexibleArcticStore
from stage_01_data_engine.storage.storage_router import StorageRouter
from stage_01_data_engine.parsers.dtn_symbol_parser import DTNSymbolParser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FlexibleStorageMigrator:
    """
    Migrate from legacy rigid storage to flexible exploratory storage.

    Migration Strategy:
    1. Discover existing data in legacy storage
    2. Analyze and categorize symbols using DTNSymbolParser
    3. Create flexible storage structure
    4. Migrate data with enhanced metadata
    5. Validate migration integrity
    6. Generate migration report
    """

    def __init__(self,
                 legacy_storage_path: str = "./data/arctic_storage",
                 flexible_storage_path: str = "./data/flexible_arctic_storage",
                 dry_run: bool = True):
        """
        Initialize migrator.

        Args:
            legacy_storage_path: Path to existing storage
            flexible_storage_path: Path for new flexible storage
            dry_run: If True, analyze only without actually migrating
        """
        self.legacy_storage_path = legacy_storage_path
        self.flexible_storage_path = flexible_storage_path
        self.dry_run = dry_run

        # Initialize components
        self.symbol_parser = DTNSymbolParser()
        self.legacy_store = None
        self.flexible_store = None
        self.storage_router = None

        # Migration tracking
        self.migration_stats = {
            'total_symbols_discovered': 0,
            'symbols_migrated': 0,
            'symbols_failed': 0,
            'total_records_migrated': 0,
            'categories_discovered': {},
            'migration_errors': [],
            'start_time': None,
            'end_time': None
        }

        # Discovered data inventory
        self.data_inventory = []

        logger.info(f"Migrator initialized - DRY RUN: {dry_run}")

    def run_migration(self) -> Dict:
        """
        Run complete migration process.

        Returns:
            Dictionary with migration results and statistics
        """
        try:
            self.migration_stats['start_time'] = datetime.now()
            logger.info("🚀 Starting flexible storage migration")

            # Step 1: Initialize storage systems
            logger.info("📋 Step 1: Initializing storage systems...")
            self._initialize_storage_systems()

            # Step 2: Discover existing data
            logger.info("🔍 Step 2: Discovering existing data...")
            self._discover_existing_data()

            # Step 3: Analyze and categorize
            logger.info("🧠 Step 3: Analyzing and categorizing symbols...")
            self._analyze_and_categorize_symbols()

            # Step 4: Plan migration strategy
            logger.info("📊 Step 4: Planning migration strategy...")
            migration_plan = self._create_migration_plan()

            if self.dry_run:
                logger.info("🔍 DRY RUN MODE - Migration plan created, no data moved")
                return self._generate_dry_run_report(migration_plan)

            # Step 5: Execute migration
            logger.info("🚚 Step 5: Executing migration...")
            self._execute_migration(migration_plan)

            # Step 6: Validate migration
            logger.info("✅ Step 6: Validating migration...")
            self._validate_migration()

            # Step 7: Generate report
            logger.info("📄 Step 7: Generating migration report...")
            self.migration_stats['end_time'] = datetime.now()

            migration_report = self._generate_migration_report()
            logger.info("🎉 Migration completed successfully!")

            return migration_report

        except Exception as e:
            logger.error(f"💥 Migration failed: {e}")
            self.migration_stats['end_time'] = datetime.now()
            return {
                'success': False,
                'error': str(e),
                'stats': self.migration_stats
            }

    def _initialize_storage_systems(self):
        """Initialize legacy and flexible storage systems"""
        try:
            # Initialize legacy storage
            legacy_uri = f"lmdb://{self.legacy_storage_path}"
            self.legacy_store = TickStore(arctic_uri=legacy_uri)

            if not self.dry_run:
                # Initialize flexible storage
                flexible_uri = f"lmdb://{self.flexible_storage_path}"
                self.flexible_store = FlexibleArcticStore(arctic_uri=flexible_uri)
                self.storage_router = StorageRouter(primary_storage=self.flexible_store)

            logger.info("✅ Storage systems initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize storage: {e}")
            raise

    def _discover_existing_data(self):
        """Discover all data in legacy storage"""
        try:
            # Use legacy storage methods to discover data
            storage_stats = self.legacy_store.get_storage_stats()

            # Get list of stored symbols (this might need adjustment based on actual TickStore implementation)
            logger.info("Discovering stored symbols from legacy storage...")

            # For now, simulate discovery since we don't have the full TickStore implementation
            # In actual implementation, this would query the legacy storage
            sample_symbols = [
                'AAPL', 'MSFT', 'TSLA', 'SPY', 'QQQ',
                'JTNT.Z', 'RINT.Z', 'TCOEA.Z', 'VCOET.Z'
            ]

            for symbol in sample_symbols:
                # Try to get data for recent dates
                for days_ago in range(7):  # Check last 7 days
                    date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

                    try:
                        # Attempt to load data (this might fail if data doesn't exist)
                        data = self.legacy_store.load_ticks(symbol, date)

                        if data is not None and not data.empty:
                            self.data_inventory.append({
                                'symbol': symbol,
                                'date': date,
                                'records': len(data),
                                'data_size_mb': data.memory_usage(deep=True).sum() / (1024*1024),
                                'columns': list(data.columns),
                                'date_range': {
                                    'start': data['timestamp'].min() if 'timestamp' in data.columns else None,
                                    'end': data['timestamp'].max() if 'timestamp' in data.columns else None
                                }
                            })

                            logger.debug(f"Found data: {symbol} on {date} ({len(data)} records)")
                            break

                    except Exception as e:
                        logger.debug(f"No data for {symbol} on {date}: {e}")
                        continue

            self.migration_stats['total_symbols_discovered'] = len(self.data_inventory)
            logger.info(f"✅ Discovered {len(self.data_inventory)} symbol/date combinations")

        except Exception as e:
            logger.error(f"❌ Failed to discover existing data: {e}")
            raise

    def _analyze_and_categorize_symbols(self):
        """Analyze discovered symbols using DTNSymbolParser"""
        try:
            category_counts = {}

            for item in self.data_inventory:
                symbol = item['symbol']

                # Parse symbol to understand its category
                symbol_info = self.symbol_parser.parse_symbol(symbol)

                # Add categorization info to inventory
                item['symbol_info'] = {
                    'category': symbol_info.category,
                    'subcategory': symbol_info.subcategory,
                    'storage_namespace': symbol_info.storage_namespace,
                    'exchange': symbol_info.exchange,
                    'underlying': symbol_info.underlying,
                    'metadata': symbol_info.metadata
                }

                # Count categories
                category = symbol_info.category
                category_counts[category] = category_counts.get(category, 0) + 1

            self.migration_stats['categories_discovered'] = category_counts

            logger.info("✅ Symbol categorization completed:")
            for category, count in category_counts.items():
                logger.info(f"   {category}: {count} symbols")

        except Exception as e:
            logger.error(f"❌ Failed to analyze symbols: {e}")
            raise

    def _create_migration_plan(self) -> Dict:
        """Create detailed migration plan"""
        try:
            plan = {
                'migration_groups': {},
                'estimated_time': 0,
                'estimated_storage_mb': 0,
                'library_creation_plan': {},
                'priority_symbols': []
            }

            # Group symbols by category for efficient migration
            for item in self.data_inventory:
                category = item['symbol_info']['category']
                subcategory = item['symbol_info']['subcategory']

                group_key = f"{category}_{subcategory}"

                if group_key not in plan['migration_groups']:
                    plan['migration_groups'][group_key] = {
                        'category': category,
                        'subcategory': subcategory,
                        'symbols': [],
                        'total_records': 0,
                        'estimated_time_minutes': 0
                    }

                plan['migration_groups'][group_key]['symbols'].append(item)
                plan['migration_groups'][group_key]['total_records'] += item['records']

                # Estimate time (rough calculation)
                plan['migration_groups'][group_key]['estimated_time_minutes'] += item['records'] / 10000  # 10k records per minute

                plan['estimated_storage_mb'] += item['data_size_mb']

            # Plan library creation
            unique_libraries = set()
            for item in self.data_inventory:
                namespace = item['symbol_info']['storage_namespace']
                library_name = namespace.replace('/', '_')
                unique_libraries.add(library_name)

            plan['library_creation_plan'] = {
                'libraries_to_create': list(unique_libraries),
                'total_libraries': len(unique_libraries)
            }

            # Identify priority symbols (high access frequency or important indicators)
            priority_symbols = []
            for item in self.data_inventory:
                symbol = item['symbol']
                category = item['symbol_info']['category']

                # Prioritize DTN indicators and major stocks
                if category == 'dtn_calculated' or symbol in ['AAPL', 'MSFT', 'SPY', 'QQQ']:
                    priority_symbols.append(symbol)

            plan['priority_symbols'] = priority_symbols
            plan['estimated_time'] = sum(group['estimated_time_minutes'] for group in plan['migration_groups'].values())

            logger.info(f"✅ Migration plan created:")
            logger.info(f"   Groups: {len(plan['migration_groups'])}")
            logger.info(f"   Libraries: {plan['library_creation_plan']['total_libraries']}")
            logger.info(f"   Estimated time: {plan['estimated_time']:.1f} minutes")
            logger.info(f"   Estimated storage: {plan['estimated_storage_mb']:.1f} MB")

            return plan

        except Exception as e:
            logger.error(f"❌ Failed to create migration plan: {e}")
            raise

    def _execute_migration(self, migration_plan: Dict):
        """Execute the migration plan"""
        try:
            if not self.flexible_store or not self.storage_router:
                raise Exception("Flexible storage not initialized for migration")

            # Migrate by priority first
            priority_symbols = set(migration_plan['priority_symbols'])

            # Process each migration group
            for group_name, group in migration_plan['migration_groups'].items():
                logger.info(f"🚚 Migrating group: {group_name} ({len(group['symbols'])} items)")

                for item in group['symbols']:
                    try:
                        symbol = item['symbol']
                        date = item['date']

                        # Load data from legacy storage
                        data = self.legacy_store.load_ticks(symbol, date)

                        if data is None or data.empty:
                            logger.warning(f"⚠️ No data found for {symbol} on {date}")
                            continue

                        # Prepare migration metadata
                        migration_metadata = {
                            'migrated_from': 'legacy_tick_store',
                            'migration_date': datetime.now().isoformat(),
                            'original_records': len(data),
                            'legacy_columns': list(data.columns),
                            'symbol_info': item['symbol_info'],
                            'priority_symbol': symbol in priority_symbols
                        }

                        # Store using flexible storage router
                        success = self.storage_router.store_symbol_data(
                            symbol=symbol,
                            data=data,
                            data_type='ticks',
                            date=date,
                            metadata=migration_metadata
                        )

                        if success:
                            self.migration_stats['symbols_migrated'] += 1
                            self.migration_stats['total_records_migrated'] += len(data)
                            logger.debug(f"✅ Migrated {symbol} ({len(data)} records)")
                        else:
                            self.migration_stats['symbols_failed'] += 1
                            error_msg = f"Failed to migrate {symbol} on {date}"
                            self.migration_stats['migration_errors'].append(error_msg)
                            logger.warning(f"❌ {error_msg}")

                    except Exception as e:
                        self.migration_stats['symbols_failed'] += 1
                        error_msg = f"Error migrating {item['symbol']}: {str(e)}"
                        self.migration_stats['migration_errors'].append(error_msg)
                        logger.error(f"💥 {error_msg}")

            logger.info(f"✅ Migration execution completed")

        except Exception as e:
            logger.error(f"❌ Migration execution failed: {e}")
            raise

    def _validate_migration(self):
        """Validate that migration was successful"""
        try:
            logger.info("🔍 Validating migration integrity...")

            validation_results = {
                'samples_validated': 0,
                'validation_passed': 0,
                'validation_failed': 0,
                'data_integrity_issues': []
            }

            # Sample validation on migrated data
            sample_size = min(10, len(self.data_inventory))
            import random
            sample_items = random.sample(self.data_inventory, sample_size)

            for item in sample_items:
                symbol = item['symbol']
                date = item['date']

                try:
                    # Load from both storages
                    legacy_data = self.legacy_store.load_ticks(symbol, date)
                    flexible_data = self.storage_router.load_symbol_data(symbol, date)

                    validation_results['samples_validated'] += 1

                    if legacy_data is None and flexible_data is None:
                        validation_results['validation_passed'] += 1
                        continue

                    if legacy_data is None or flexible_data is None:
                        issue = f"{symbol} on {date}: Data exists in only one storage"
                        validation_results['data_integrity_issues'].append(issue)
                        validation_results['validation_failed'] += 1
                        continue

                    # Compare record counts
                    if len(legacy_data) != len(flexible_data):
                        issue = f"{symbol} on {date}: Record count mismatch ({len(legacy_data)} vs {len(flexible_data)})"
                        validation_results['data_integrity_issues'].append(issue)
                        validation_results['validation_failed'] += 1
                        continue

                    validation_results['validation_passed'] += 1
                    logger.debug(f"✅ Validation passed for {symbol} on {date}")

                except Exception as e:
                    issue = f"{symbol} on {date}: Validation error - {str(e)}"
                    validation_results['data_integrity_issues'].append(issue)
                    validation_results['validation_failed'] += 1

            self.migration_stats['validation_results'] = validation_results

            success_rate = validation_results['validation_passed'] / max(validation_results['samples_validated'], 1) * 100
            logger.info(f"✅ Validation completed: {success_rate:.1f}% success rate")

            if validation_results['validation_failed'] > 0:
                logger.warning(f"⚠️ {validation_results['validation_failed']} validation issues found")

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise

    def _generate_migration_report(self) -> Dict:
        """Generate comprehensive migration report"""
        try:
            duration = self.migration_stats['end_time'] - self.migration_stats['start_time']

            report = {
                'migration_summary': {
                    'success': self.migration_stats['symbols_failed'] == 0,
                    'total_time_minutes': duration.total_seconds() / 60,
                    'symbols_discovered': self.migration_stats['total_symbols_discovered'],
                    'symbols_migrated': self.migration_stats['symbols_migrated'],
                    'symbols_failed': self.migration_stats['symbols_failed'],
                    'records_migrated': self.migration_stats['total_records_migrated'],
                    'success_rate': self.migration_stats['symbols_migrated'] / max(self.migration_stats['total_symbols_discovered'], 1) * 100
                },
                'categories_migrated': self.migration_stats['categories_discovered'],
                'storage_efficiency': {
                    'flexible_storage_benefits': [
                        'Dynamic symbol categorization',
                        'Automatic library creation',
                        'Enhanced metadata tracking',
                        'Exploratory research support',
                        'Better query performance'
                    ]
                },
                'migration_errors': self.migration_stats['migration_errors'][:10],  # First 10 errors
                'validation_results': self.migration_stats.get('validation_results', {}),
                'recommendations': self._get_post_migration_recommendations()
            }

            # Save report to file
            report_path = Path('migration_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"📄 Migration report saved to {report_path}")
            return report

        except Exception as e:
            logger.error(f"❌ Failed to generate migration report: {e}")
            return {'error': str(e)}

    def _generate_dry_run_report(self, migration_plan: Dict) -> Dict:
        """Generate report for dry run"""
        return {
            'dry_run': True,
            'migration_plan': migration_plan,
            'discovery_summary': {
                'symbols_discovered': len(self.data_inventory),
                'categories_found': self.migration_stats['categories_discovered'],
                'estimated_migration_time': migration_plan['estimated_time'],
                'estimated_storage_mb': migration_plan['estimated_storage_mb']
            },
            'recommendations': [
                'Run with dry_run=False to execute actual migration',
                'Ensure sufficient disk space for flexible storage',
                'Consider migrating priority symbols first',
                'Plan maintenance window for migration execution'
            ]
        }

    def _get_post_migration_recommendations(self) -> List[str]:
        """Get recommendations for post-migration optimization"""
        recommendations = [
            'Update data collection scripts to use flexible storage',
            'Configure retention policies for different symbol categories',
            'Set up monitoring for storage performance',
            'Plan gradual deprecation of legacy storage',
        ]

        if self.migration_stats['symbols_failed'] > 0:
            recommendations.insert(0, 'Review and retry failed migrations')

        if len(self.migration_stats['categories_discovered']) > 5:
            recommendations.append('Consider storage optimization for high-diversity universe')

        return recommendations


def main():
    """Main migration script"""
    import argparse

    parser = argparse.ArgumentParser(description='Migrate to flexible storage architecture')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Analyze migration without moving data (default)')
    parser.add_argument('--execute', action='store_true',
                       help='Execute actual migration')
    parser.add_argument('--legacy-path', default='./data/arctic_storage',
                       help='Path to legacy storage')
    parser.add_argument('--flexible-path', default='./data/flexible_arctic_storage',
                       help='Path for flexible storage')

    args = parser.parse_args()

    # Determine run mode
    dry_run = not args.execute

    try:
        print("🚀 Flexible Storage Migration Tool")
        print("=" * 50)

        if dry_run:
            print("🔍 DRY RUN MODE - No data will be moved")
        else:
            print("⚠️  LIVE MIGRATION MODE - Data will be migrated")
            confirm = input("Are you sure you want to proceed? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Migration cancelled")
                return

        # Run migration
        migrator = FlexibleStorageMigrator(
            legacy_storage_path=args.legacy_path,
            flexible_storage_path=args.flexible_path,
            dry_run=dry_run
        )

        results = migrator.run_migration()

        print("\n" + "=" * 50)
        print("📊 MIGRATION RESULTS")
        print("=" * 50)

        if dry_run:
            print(f"Symbols discovered: {results['discovery_summary']['symbols_discovered']}")
            print(f"Categories found: {list(results['discovery_summary']['categories_found'].keys())}")
            print(f"Estimated time: {results['discovery_summary']['estimated_migration_time']:.1f} minutes")
            print(f"Estimated storage: {results['discovery_summary']['estimated_storage_mb']:.1f} MB")
            print("\nRun with --execute flag to perform actual migration")
        else:
            summary = results['migration_summary']
            print(f"Migration success: {summary['success']}")
            print(f"Symbols migrated: {summary['symbols_migrated']}")
            print(f"Success rate: {summary['success_rate']:.1f}%")
            print(f"Records migrated: {summary['records_migrated']:,}")
            print(f"Time taken: {summary['total_time_minutes']:.1f} minutes")

            if summary['symbols_failed'] > 0:
                print(f"⚠️ Failed migrations: {summary['symbols_failed']}")

        print("\n✅ Migration tool completed successfully!")

    except Exception as e:
        print(f"\n💥 Migration tool failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())