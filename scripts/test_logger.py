from utils.logger import get_logger


logger = get_logger(__name__)


def main():

    logger.info("Logger initialized successfully.")

    logger.info("SBI Insurance RAG project started.")

    logger.warning("This is a test warning.")


if __name__ == "__main__":
    main()