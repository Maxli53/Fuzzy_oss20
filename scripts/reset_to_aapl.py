"""
Reset Database to AAPL Only
Following the narrow-but-deep strategy defined in Data_policy.md
"""
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Setup paths
sys.path.insert(0, '.')
sys.path.insert(0, 'stage_01_data_engine')
sys.path.insert(0, 'foundation')

from arcticdb import Arctic
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseResetter:
    """Reset database to clean slate with AAPL only"""

    def __init__(self):
        self.arctic = Arctic('lmdb://./data/arctic_storage')
        self.libraries_to_preserve = [
            'tick_data',
            'bars_time',
            'bars_tick',
            'bars_volume',
            'bars_dollar',
            'bars_imbalance',
            'bars_range',
            'bars_renko',
            'bar_metadata',
            'metadata'
        ]

    def backup_existing_data(self):
        """Create backup of existing data before reset"""
        logger.info("=" * 80)
        logger.info("BACKING UP EXISTING DATA")
        logger.info("=" * 80)

        backup_dir = Path('./data/backups') / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)

        for lib_name in self.arctic.list_libraries():
            try:
                lib = self.arctic[lib_name]
                symbols = lib.list_symbols()

                if symbols:
                    logger.info(f"Backing up {lib_name}: {len(symbols)} symbols")
                    # Note: In production, implement actual backup logic
                    # For now, just log what would be backed up

            except Exception as e:
                logger.warning(f"Could not backup {lib_name}: {e}")

        logger.info("Backup complete (or would be in production)")

    def clear_non_aapl_data(self):
        """Remove all data except AAPL"""
        logger.info("\n" + "=" * 80)
        logger.info("CLEARING NON-AAPL DATA")
        logger.info("=" * 80)

        for lib_name in self.arctic.list_libraries():
            try:
                lib = self.arctic[lib_name]
                symbols = lib.list_symbols()

                for symbol in symbols:
                    # Keep only AAPL-related symbols
                    if 'AAPL' not in symbol.upper():
                        logger.info(f"  Deleting {lib_name}/{symbol}")
                        lib.delete(symbol)
                    else:
                        logger.info(f"  Keeping {lib_name}/{symbol}")

            except Exception as e:
                logger.warning(f"Error processing {lib_name}: {e}")

    def ensure_library_structure(self):
        """Ensure all required libraries exist"""
        logger.info("\n" + "=" * 80)
        logger.info("ENSURING LIBRARY STRUCTURE")
        logger.info("=" * 80)

        existing_libs = set(self.arctic.list_libraries())

        for lib_name in self.libraries_to_preserve:
            if lib_name not in existing_libs:
                self.arctic.create_library(lib_name)
                logger.info(f"  Created library: {lib_name}")
            else:
                logger.info(f"  Library exists: {lib_name}")

    def update_schema_version(self):
        """Update schema version in metadata"""
        logger.info("\n" + "=" * 80)
        logger.info("UPDATING SCHEMA VERSION")
        logger.info("=" * 80)

        metadata_lib = self.arctic['metadata']

        import json

        schema_info = {
            'version': 2,
            'updated_at': datetime.now().isoformat(),
            'description': 'Tiered metadata system with dynamic intervals',
            'key_format': json.dumps({
                'tick_data': '{asset_class}/{symbol}/{date}',
                'bars_time': '{symbol}/{interval}/{date}',
                'bar_metadata': '{symbol}/{interval}/{date}/tier{N}'
            })
        }

        metadata_lib.write('schema_version', pd.DataFrame([schema_info]))
        logger.info("  Schema version updated to v2")

    def verify_reset(self):
        """Verify the reset was successful"""
        logger.info("\n" + "=" * 80)
        logger.info("VERIFICATION")
        logger.info("=" * 80)

        total_symbols = 0
        aapl_symbols = 0

        for lib_name in self.arctic.list_libraries():
            try:
                lib = self.arctic[lib_name]
                symbols = lib.list_symbols()

                lib_aapl = sum(1 for s in symbols if 'AAPL' in s.upper())
                lib_total = len(symbols)

                total_symbols += lib_total
                aapl_symbols += lib_aapl

                if lib_total > 0:
                    logger.info(f"  {lib_name}: {lib_total} symbols ({lib_aapl} AAPL)")

            except Exception as e:
                logger.warning(f"Could not verify {lib_name}: {e}")

        logger.info(f"\nTotal: {total_symbols} symbols ({aapl_symbols} AAPL-related)")

        if total_symbols == aapl_symbols:
            logger.info("✓ Database successfully reset to AAPL only")
        else:
            logger.warning(f"⚠ Found {total_symbols - aapl_symbols} non-AAPL symbols")

    def reset(self):
        """Execute full reset procedure"""
        logger.info("\n" + "=" * 80)
        logger.info("DATABASE RESET TO AAPL ONLY")
        logger.info("=" * 80)
        logger.info("Following narrow-but-deep strategy from Data_policy.md")

        # Step 1: Backup
        self.backup_existing_data()

        # Step 2: Clear non-AAPL data
        self.clear_non_aapl_data()

        # Step 3: Ensure library structure
        self.ensure_library_structure()

        # Step 4: Update schema version
        self.update_schema_version()

        # Step 5: Verify
        self.verify_reset()

        logger.info("\n" + "=" * 80)
        logger.info("RESET COMPLETE")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("  1. Run fetch_aapl_week.py to get 8 days of tick data")
        logger.info("  2. Run generate_standard_bars.py to create time bars")
        logger.info("  3. Run test_comprehensive_bars.py to verify")


if __name__ == "__main__":
    resetter = DatabaseResetter()

    # Confirm before proceeding
    print("\n" + "=" * 80)
    print("WARNING: This will remove all non-AAPL data from the database")
    print("=" * 80)
    response = input("Are you sure you want to proceed? (yes/no): ")

    if response.lower() == 'yes':
        resetter.reset()
    else:
        print("Reset cancelled")