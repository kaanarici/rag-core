import sys

import rag_core


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: ask_folder <folder> <question>")
    folder, question = sys.argv[1], " ".join(sys.argv[2:])
    with rag_core.index(folder) as idx:
        print(idx.context(question))


if __name__ == "__main__":
    main()
