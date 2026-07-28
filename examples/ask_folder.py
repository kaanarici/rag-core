import sys

from rag_core.easy import index


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: ask_folder <folder> <question>")
    folder, question = sys.argv[1], " ".join(sys.argv[2:])
    with index(folder) as idx:
        print(idx.context(question))


if __name__ == "__main__":
    main()
