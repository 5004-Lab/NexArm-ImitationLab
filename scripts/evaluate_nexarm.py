import sys

from imitation_lab.cli import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "offline":
        arguments[0] = "evaluate"
    main(arguments)
