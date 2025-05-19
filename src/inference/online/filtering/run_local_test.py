import yaml
import logging
import os
from datetime import datetime
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement, BatchStatement
from cassandra.util import uuid_from_time

# Assuming filter_service.py is in the same directory or accessible via PYTHONPATH
from filter_service import FilterService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'filtering_service_config.yaml')


def load_config(config_path):
    """Loads YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

def create_schema_and_insert_test_data(config):
    """
    Connects to ScyllaDB, creates keyspace/table if they don't exist,
    and inserts sample test data from the configuration.
    """
    scylla_contact_points = config.get('scylla_contact_points', ["127.0.0.1"])
    scylla_port = config.get('scylla_port', 9042)
    keyspace_name = config.get('scylla_keyspace', "recsys_poc_interactions")
    sample_data = config.get('sample_test_data', [])

    cluster = None
    try:
        cluster = Cluster(contact_points=scylla_contact_points, port=scylla_port)
        session = cluster.connect()

        # Create Keyspace
        session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {keyspace_name}
            WITH REPLICATION = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
        """)
        logger.info(f"Keyspace '{keyspace_name}' ensured.")
        session.set_keyspace(keyspace_name)

        # Create Table
        session.execute("""
            CREATE TABLE IF NOT EXISTS user_post_interactions (
                user_id TEXT,
                post_id TEXT,
                interaction_type TEXT,
                interaction_timestamp TIMESTAMP,
                PRIMARY KEY ((user_id), post_id, interaction_type)
            );
        """)
        logger.info("Table 'user_post_interactions' ensured.")

        # Insert Sample Data
        if sample_data:
            # Check if data for a specific user already exists to avoid re-inserting every time.
            # This is a simple check; more robust checks might be needed for complex scenarios.
            # For this PoC, we'll insert if the table is empty or for specific test users if needed.
            # A more robust way would be to check for specific user_id/post_id/interaction_type combos.

            # Let's clear out data for test users before inserting to ensure a clean state for the test.
            # This is specific to this test script.
            test_user_ids_in_sample = list(set(item['user_id'] for item in sample_data))
            if test_user_ids_in_sample:
                logger.info(f"Clearing existing data for test users: {test_user_ids_in_sample} before inserting new samples.")
                for test_user_id in test_user_ids_in_sample:
                    # This delete is broad for the user; be cautious in production.
                    # For testing, it's okay to ensure a clean slate for the sample data.
                    # A more precise delete would include post_id and interaction_type if needed.
                    session.execute(f"DELETE FROM user_post_interactions WHERE user_id = %s", (test_user_id,))


            prepared_insert = session.prepare(
                "INSERT INTO user_post_interactions (user_id, post_id, interaction_type, interaction_timestamp) "
                "VALUES (?, ?, ?, ?)"
            )
            batch = BatchStatement()
            insert_count = 0
            for item in sample_data:
                # cassandra-driver expects datetime objects for timestamp columns
                # 'now()' in YAML was a placeholder; use current time or parse if a specific timestamp string is provided.
                # For simplicity, using current time for each sample record.
                # For more control, you could parse a string timestamp from YAML.
                timestamp = datetime.utcnow() # Use a consistent timestamp for the batch or individual ones
                batch.add(prepared_insert, (item['user_id'], item['post_id'], item['interaction_type'], timestamp))
                insert_count +=1
            
            if insert_count > 0:
                session.execute(batch)
                logger.info(f"Inserted/Updated {insert_count} sample interaction records.")
            else:
                logger.info("No sample data to insert.")

    except Exception as e:
        logger.error(f"ScyllaDB setup or data insertion failed: {e}")
        # Depending on requirements, you might want to re-raise or handle gracefully
    finally:
        if cluster:
            cluster.shutdown()

def run_test():
    logger.info("Starting Filtering Service local test...")
    config = load_config(CONFIG_PATH)

    # 1. Setup ScyllaDB and insert test data (idempotent)
    # This ensures the schema and sample data are ready for the test.
    create_schema_and_insert_test_data(config)

    # 2. Instantiate FilterService
    filter_service = None
    try:
        filter_service = FilterService(
            scylla_contact_points=config['scylla_contact_points'],
            scylla_keyspace=config['scylla_keyspace'],
            scylla_port=config.get('scylla_port', 9042) # Use get for optional port
        )

        # 3. Simulate candidate post_ids
        candidate_posts_all = ["post_100", "post_101", "post_102", "post_103", "post_104", "post_105", "post_201", "post_202"]

        # Test Case 1: User "user_filter_test_1"
        # From config:
        # - user_filter_test_1, post_101, seen
        # - user_filter_test_1, post_102, disliked
        # - user_filter_test_1, post_103, seen
        test_user_1 = "user_filter_test_1"
        logger.info(f"\n--- Testing with user: {test_user_1} ---")
        logger.info(f"Original candidate posts: {candidate_posts_all}")
        
        filtered_posts_1 = filter_service.filter_candidates(
            user_id=test_user_1,
            candidate_post_ids=candidate_posts_all
        )
        logger.info(f"Filtered posts for {test_user_1}: {filtered_posts_1}")
        # Expected: ["post_100", "post_104", "post_105", "post_201", "post_202"] (post_101, post_102, post_103 removed)
        
        # Test Case 2: User "user_filter_test_2"
        # From config:
        # - user_filter_test_2, post_201, reported
        test_user_2 = "user_filter_test_2"
        logger.info(f"\n--- Testing with user: {test_user_2} ---")
        logger.info(f"Original candidate posts: {candidate_posts_all}")

        filtered_posts_2 = filter_service.filter_candidates(
            user_id=test_user_2,
            candidate_post_ids=candidate_posts_all
        )
        logger.info(f"Filtered posts for {test_user_2}: {filtered_posts_2}")
        # Expected: ["post_100", "post_101", "post_102", "post_103", "post_104", "post_105", "post_202"] (post_201 removed)

        # Test Case 3: User with no interactions in DB
        test_user_3 = "user_with_no_interactions"
        logger.info(f"\n--- Testing with user: {test_user_3} (no prior interactions) ---")
        logger.info(f"Original candidate posts: {candidate_posts_all}")
        
        filtered_posts_3 = filter_service.filter_candidates(
            user_id=test_user_3,
            candidate_post_ids=candidate_posts_all
        )
        logger.info(f"Filtered posts for {test_user_3}: {filtered_posts_3}")
        # Expected: All posts should remain, same as candidate_posts_all

        # Test Case 4: Empty candidate list
        logger.info(f"\n--- Testing with empty candidate list for user: {test_user_1} ---")
        filtered_empty = filter_service.filter_candidates(
            user_id=test_user_1,
            candidate_post_ids=[]
        )
        logger.info(f"Filtered posts for empty list: {filtered_empty}")
        # Expected: []


    except Exception as e:
        logger.error(f"An error occurred during the FilterService test run: {e}", exc_info=True)
    finally:
        if filter_service:
            logger.info("Shutting down FilterService.")
            filter_service.shutdown()
        logger.info("Filtering Service local test finished.")

if __name__ == "__main__":
    # Ensure ScyllaDB is running before executing this script.
    # Example: docker run -p 9042:9042 --name scylla -d scylladb/scylla
    # You might need to wait a bit for ScyllaDB to initialize fully after starting.
    run_test()