import argparse
import yaml
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming src is in PYTHONPATH or the script is run from the project root.
# Adjust imports if necessary based on your execution environment.
from src.inference.online.candidate_generation.predictor import UserTowerPredictor
from src.inference.online.candidate_generation.transformer import CandidateGenerationTransformer
from src.inference.online.filtering.filter_service import FilterService
from src.inference.online.ranking.predictor import RankingModelPredictor
from src.inference.online.ranking.transformer import RankingServiceTransformer
from src.inference.online.re_ranking.re_rank_service import ReRankService

class OnlineInferencePipeline:
    def __init__(self, 
                 candidate_gen_config_path: str, 
                 filtering_config_path: str, 
                 ranking_config_path: str, 
                 re_ranking_config_path: str):
        
        logging.info("Initializing OnlineInferencePipeline...")

        # Load configurations
        with open(candidate_gen_config_path, 'r') as f:
            cg_config = yaml.safe_load(f)
        with open(filtering_config_path, 'r') as f:
            filter_config = yaml.safe_load(f)
        with open(ranking_config_path, 'r') as f:
            rank_config = yaml.safe_load(f)
        with open(re_ranking_config_path, 'r') as f:
            re_rank_config = yaml.safe_load(f)

        # Instantiate services
        # Candidate Generation
        user_tower_predictor = UserTowerPredictor(
            model_uri=cg_config['user_tower_model_uri'],
            user_features=cg_config['user_features']
        )
        self.candidate_generation_transformer = CandidateGenerationTransformer(
            config=cg_config,
            user_tower_predictor=user_tower_predictor
        )
        logging.info("Candidate Generation service initialized.")

        # Filtering
        self.filter_service = FilterService(config=filter_config)
        logging.info("Filtering service initialized.")

        # Ranking
        ranking_model_predictor = RankingModelPredictor(
            model_uri=rank_config['ranking_model_uri'],
            user_features=rank_config['user_features'],
            post_features=rank_config['post_features']
        )
        self.ranking_service_transformer = RankingServiceTransformer(
            config=rank_config,
            ranking_model_predictor=ranking_model_predictor
        )
        logging.info("Ranking service initialized.")

        # Re-ranking
        self.re_rank_service = ReRankService(config=re_rank_config)
        logging.info("Re-ranking service initialized.")
        
        logging.info("OnlineInferencePipeline initialized successfully.")

    def get_recommendations(self, 
                            user_id: str, 
                            preferred_category_ids: List[int], 
                            top_n_candidates: int = 100, 
                            num_final_recommendations: int = 20) -> List[str]:
        logging.info(f"Starting recommendation generation for user_id: {user_id}")

        # Stage 1: Candidate Generation
        cg_request_data = {
            "user_id": user_id,
            "preferred_category_ids": preferred_category_ids,
            "top_n_candidates": top_n_candidates
        }
        logging.info(f"Candidate Generation - Request: {cg_request_data}")
        candidate_post_ids = self.candidate_generation_transformer.generate_candidates(cg_request_data)
        logging.info(f"Candidate Generation - Generated {len(candidate_post_ids)} candidates: {candidate_post_ids[:10]}...") # Log first 10

        if not candidate_post_ids:
            logging.warning("Candidate Generation returned no candidates.")
            return []

        # Stage 2: Filtering
        logging.info(f"Filtering - Input candidates: {len(candidate_post_ids)}")
        filtered_post_ids = self.filter_service.filter_candidates(user_id, candidate_post_ids)
        logging.info(f"Filtering - Filtered to {len(filtered_post_ids)} candidates: {filtered_post_ids[:10]}...") # Log first 10

        if not filtered_post_ids:
            logging.warning("Filtering returned no candidates after applying filters.")
            return []

        # Stage 3: Ranking
        ranking_request_data = {
            "user_id": user_id,
            "post_ids": filtered_post_ids
        }
        logging.info(f"Ranking - Request for {len(filtered_post_ids)} candidates.")
        scored_posts = self.ranking_service_transformer.rank_candidates(ranking_request_data)
        logging.info(f"Ranking - Scored {len(scored_posts)} candidates. Top 5: {scored_posts[:5]}")

        if not scored_posts:
            logging.warning("Ranking returned no scored posts.")
            return []

        # Stage 4: Re-ranking & Presentation
        logging.info(f"Re-ranking - Input {len(scored_posts)} scored posts, requesting {num_final_recommendations} final recommendations.")
        final_recommendations = self.re_rank_service.re_rank_and_present(scored_posts, num_final_recommendations)
        logging.info(f"Re-ranking - Final {len(final_recommendations)} recommendations: {final_recommendations}")

        return final_recommendations

    def shutdown(self):
        logging.info("Shutting down OnlineInferencePipeline...")
        if self.filter_service:
            self.filter_service.shutdown()
        # Add shutdown for other services if they have persistent connections (e.g., Milvus client in CG if not managed per-request)
        # For this PoC, candidate_generation_transformer's Milvus client is instantiated per call in generate_candidates.
        logging.info("OnlineInferencePipeline shut down.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full online inference pipeline.")
    parser.add_argument("--user_id", type=str, required=True, help="ID of the user to get recommendations for.")
    parser.add_argument("--categories", type=int, nargs='+', required=True, help="Space-separated list of preferred category IDs.")
    parser.add_argument("--top_n_candidates", type=int, default=100, help="Number of initial candidates to generate.")
    parser.add_argument("--num_final_recommendations", type=int, default=20, help="Number of final recommendations to return.")
    
    parser.add_argument("--cg_config", type=str, default="config/candidate_generation_service_config.yaml", help="Path to Candidate Generation service config YAML.")
    parser.add_argument("--filter_config", type=str, default="config/filtering_service_config.yaml", help="Path to Filtering service config YAML.")
    parser.add_argument("--rank_config", type=str, default="config/ranking_service_config.yaml", help="Path to Ranking service config YAML.")
    parser.add_argument("--re_rank_config", type=str, default="config/re_ranking_service_config.yaml", help="Path to Re-ranking service config YAML.")

    args = parser.parse_args()

    pipeline = None
    try:
        pipeline = OnlineInferencePipeline(
            candidate_gen_config_path=args.cg_config,
            filtering_config_path=args.filter_config,
            ranking_config_path=args.rank_config,
            re_ranking_config_path=args.re_rank_config
        )

        recommendations = pipeline.get_recommendations(
            user_id=args.user_id,
            preferred_category_ids=args.categories,
            top_n_candidates=args.top_n_candidates,
            num_final_recommendations=args.num_final_recommendations
        )

        print("\nFinal Recommended Post IDs:")
        if recommendations:
            for post_id in recommendations:
                print(post_id)
        else:
            print("No recommendations generated.")

    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        if pipeline:
            pipeline.shutdown()