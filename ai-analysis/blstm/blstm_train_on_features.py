import blstm
import logging
import blstm_training

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    print(f"\n{'=' * 50}")
    print("Component 1 BiLSTM C1 Training Starting")
    print(f"{'=' * 50}")

    logger.info("Starting Component 1 BLSTM Training with Essay Features")
    logger.info("=" * 70)

    # Setup
    device = blstm.get_device("auto")
    blstm.set_seed(42)
    logger.info(f"Using device: {device}")

    try:
        blstm_training.train_on_features(device)

        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    main()
