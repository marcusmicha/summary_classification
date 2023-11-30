import argparse
from src.attacher import Attacher
from src.utils import load_config
from rich.console import Console
console = Console()

def main(args: dict):
    config = load_config(args['config_path'])
    attach = Attacher(**args, **config)
    attach()
    res = attach.sort()
    res.to_csv(config['final_file_path'])
    console.print(f'[green] Final csv file saved under {config["final_file_path"]} [/green]')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dialogues_path', help='dialogues file path', type=str, default='data/dialogues.csv')
    parser.add_argument('-S', '--summaries_path', help='summaries file path', type=str, default='data/summary_pieces.csv')
    parser.add_argument('-R', '--references_path', help='references file path', type=str, default='data/reference.csv')
    parser.add_argument('-C', '--config_path', help='config file path', type=str, default='config.yml')

    args = parser.parse_args()
    main(vars(args))