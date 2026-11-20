import sys

from imitation_lab.cli import main

if __name__ == "__main__":
    main(["convert", *sys.argv[1:]])
