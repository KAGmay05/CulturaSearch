import argparse
import sys


def run_crawler():
    from crawler.run_crawler import main as crawler_main

    crawler_main()


def run_scraper():
    from scraper.run_scraper import main as scraper_main

    scraper_main()


def run_rag():
    from rag_module.run_rag import main as rag_main

    rag_main()


def run_full():
    run_crawler()
    run_scraper()
    run_rag()


def build_parser():
    parser = argparse.ArgumentParser(description="CulturaSearch - runner general")
    parser.add_argument(
        "command",
        choices=["crawl", "scrape", "rag", "full"],
        nargs="?",
        default="full",
        help="Subcomando a ejecutar",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "crawl": run_crawler,
        "scrape": run_scraper,
        "rag": run_rag,
        "full": run_full,
    }

    try:
        commands[args.command]()
    except KeyboardInterrupt:
        print("\n✋ Operación interrumpida por el usuario")
        sys.exit(0)


if __name__ == "__main__":
    main()