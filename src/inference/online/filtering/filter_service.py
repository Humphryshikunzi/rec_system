import logging
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

logger = logging.getLogger(__name__)

class FilterService:
    def __init__(self, scylla_contact_points: list, scylla_keyspace: str, scylla_port: int = 9042):
        """
        Initializes the FilterService with a connection to ScyllaDB.

        Args:
            scylla_contact_points (list): List of ScyllaDB contact points (IP addresses or hostnames).
            scylla_keyspace (str): The keyspace to use in ScyllaDB.
            scylla_port (int): The port number for ScyllaDB.
        """
        try:
            self.cluster = Cluster(
                contact_points=scylla_contact_points,
                port=scylla_port,
                protocol_version=4 # Re-adding explicit protocol version
            )
            self.session = self.cluster.connect()
            self.keyspace = scylla_keyspace
            self.session.set_keyspace(self.keyspace)
            logger.info(f"Successfully connected to ScyllaDB at {scylla_contact_points} and set keyspace to {self.keyspace}")
        except Exception as e:
            logger.error(f"Failed to connect to ScyllaDB or set keyspace: {e}")
            raise

    def _get_user_interactions(self, user_id: str, post_ids: list, interaction_types: list) -> set:
        """
        Queries ScyllaDB for the given user_id and post_ids to find any interactions
        matching interaction_types.

        Args:
            user_id (str): The ID of the user.
            post_ids (list): A list of post_ids to check for interactions.
            interaction_types (list): A list of interaction types (e.g., ['seen', 'disliked']).

        Returns:
            set: A set of post_ids that have matching interactions.
        """
        if not post_ids or not interaction_types:
            return set()

        try:
            if not post_ids or not interaction_types: # Ensure lists are not empty before creating placeholders
                return set()

            post_id_placeholders = ', '.join(['?'] * len(post_ids))
            interaction_type_placeholders = ', '.join(['?'] * len(interaction_types))

            query_string = (
                f"SELECT post_id FROM {self.keyspace}.user_post_interactions "
                f"WHERE user_id = ? AND post_id IN ({post_id_placeholders}) AND interaction_type IN ({interaction_type_placeholders})"
            )
            
            params = [user_id] + list(post_ids) + list(interaction_types)
            
            # Use a prepared statement
            prepared_statement = self.session.prepare(query_string)
            rows = self.session.execute(prepared_statement, params)
            interacted_post_ids = {row.post_id for row in rows}
            logger.debug(f"User {user_id} interacted with posts: {interacted_post_ids} for types {interaction_types}")
            return interacted_post_ids
        except Exception as e:
            logger.error(f"Error querying ScyllaDB for user interactions: {e}")
            return set() # Return empty set on error to avoid breaking the filtering flow

    def filter_candidates(self, user_id: str, candidate_post_ids: list) -> list:
        """
        Filters candidate post_ids by removing those the user has already interacted with
        (seen, disliked, reported).

        Args:
            user_id (str): The ID of the user.
            candidate_post_ids (list): A list of candidate post_ids.

        Returns:
            list: A new list containing only post_ids from candidate_post_ids
                  that are not in the interacted set.
        """
        if not candidate_post_ids:
            return []

        interaction_types_to_filter = ['seen', 'disliked', 'reported']
        
        interacted_posts = self._get_user_interactions(
            user_id=user_id,
            post_ids=candidate_post_ids,
            interaction_types=interaction_types_to_filter
        )

        filtered_candidates = [
            post_id for post_id in candidate_post_ids if post_id not in interacted_posts
        ]
        
        logger.info(f"Original candidates for user {user_id}: {len(candidate_post_ids)}. "
                    f"Filtered candidates: {len(filtered_candidates)}. "
                    f"Removed: {len(interacted_posts)} posts.")
        return filtered_candidates

    def shutdown(self):
        """
        Properly closes the ScyllaDB connection.
        """
        if self.cluster:
            logger.info("Shutting down ScyllaDB connection.")
            self.cluster.shutdown()

if __name__ == '__main__':
    # Basic test setup (replace with actual configuration and data for real testing)
    logging.basicConfig(level=logging.INFO)
    
    # This is a placeholder for where you'd normally load config
    SCYLLA_CONTACT_POINTS = ["127.0.0.1"]
    SCYLLA_KEYSPACE = "recsys_poc_interactions" # Ensure this keyspace and table exist

    # --- IMPORTANT: Manual ScyllaDB Setup Required for this basic test ---
    # 1. Start ScyllaDB (e.g., using Docker: docker run -p 9042:9042 --name scylla -d scylladb/scylla)
    # 2. Connect via cqlsh: docker exec -it scylla cqlsh
    # 3. Create Keyspace and Table:
    #    CREATE KEYSPACE IF NOT EXISTS recsys_poc_interactions WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1};
    #    USE recsys_poc_interactions;
    #    CREATE TABLE IF NOT EXISTS user_post_interactions (
    #        user_id TEXT,
    #        post_id TEXT,
    #        interaction_type TEXT,
    #        interaction_timestamp TIMESTAMP,
    #        PRIMARY KEY ((user_id), post_id, interaction_type)
    #    );
    # 4. Insert some sample data for 'user1':
    #    INSERT INTO user_post_interactions (user_id, post_id, interaction_type, interaction_timestamp) VALUES ('user1', 'post_A', 'seen', toTimestamp(now()));
    #    INSERT INTO user_post_interactions (user_id, post_id, interaction_type, interaction_timestamp) VALUES ('user1', 'post_C', 'disliked', toTimestamp(now()));
    # ---------------------------------------------------------------------

    try:
        filter_service = FilterService(
            scylla_contact_points=SCYLLA_CONTACT_POINTS,
            scylla_keyspace=SCYLLA_KEYSPACE
        )

        test_user_id = "user1"
        candidate_posts = ["post_A", "post_B", "post_C", "post_D"]

        print(f"Original candidate posts for {test_user_id}: {candidate_posts}")
        
        filtered_posts = filter_service.filter_candidates(
            user_id=test_user_id,
            candidate_post_ids=candidate_posts
        )
        print(f"Filtered posts for {test_user_id}: {filtered_posts}")
        # Expected for 'user1' with sample data: ['post_B', 'post_D']

        # Test with a user with no interactions
        test_user_no_interactions = "user_no_data"
        print(f"Original candidate posts for {test_user_no_interactions}: {candidate_posts}")
        filtered_posts_no_interactions = filter_service.filter_candidates(
            user_id=test_user_no_interactions,
            candidate_post_ids=candidate_posts
        )
        print(f"Filtered posts for {test_user_no_interactions}: {filtered_posts_no_interactions}")
        # Expected: ['post_A', 'post_B', 'post_C', 'post_D']


    except Exception as e:
        logger.error(f"An error occurred during the basic test: {e}")
    finally:
        if 'filter_service' in locals() and filter_service:
            filter_service.shutdown()