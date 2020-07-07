from pathlib import Path

from glom import glom
import yaml
import defopt


def main(glom_pattern: str, path: Path):
    with open(path, 'r') as f:
        data = yaml.load(f)
    print(glom(data, glom_pattern))


def cli():
    defopt.run(main)

if __name__ == "__main__":
    cli()
