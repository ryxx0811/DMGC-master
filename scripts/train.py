"""Train or evaluate DMGC from the repository root.

Example:
    python -m scripts.train --task spatial --mode train
"""

import logging
from pathlib import Path

import torch

from dmgc.config import parse_args
from dmgc.data import GraphDataLoader
from dmgc.models import DMGC
from dmgc.training import GraphTrainer
from dmgc.utils import move_to_cpu, set_seed


EXPERIMENTS = {
    'single_cell': {
        'n_clusters': [15, 15, 15, 15],
        'dims': [[1884, 512, 128]] * 4,
    },
    'heart': {
        'n_clusters': [8, 8, 8, 6, 6],
        'dims': [[2000, 512, 128]] * 5,
    },
    'spatial': {
        'n_clusters': [6, 6, 6, 6, 6, 6],
        'dims': [[500, 128, 64]] * 6,
    },
}


def save_outputs(output_dir, h, a, zs, cs, qs):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in {'h': h, 'a': a, 'zs': zs, 'Cs': cs, 'qs': qs}.items():
        torch.save(move_to_cpu(value), output_dir / f'{name}.pt')


def main():
    args = parse_args()
    set_seed(42)

    experiment = EXPERIMENTS[args.task]
    n_clusters = experiment['n_clusters']
    dims = experiment['dims']
    n_networks = len(n_clusters)
    output_dir = Path(args.output_dir) / args.task
    checkpoint = Path(args.checkpoint) if args.checkpoint else output_dir / 'model.pt'

    data_loader = GraphDataLoader(
        args.task, n_networks, n_clusters, dims, args.centroid_dim, args.beta, args.output_dir,
    )
    xs, cnets, cnets_masks, u_is, u_js, u_labels, b = data_loader.get_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DMGC(n_networks, dims, args.centroid_dim, n_clusters).to(device)
    xs = [x.to(device) for x in xs]
    trainer = GraphTrainer(model, n_networks, n_clusters, u_is, u_js, u_labels, b, cnets, cnets_masks, args)

    if args.mode == 'evaluate':
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        for epoch in range(1, args.epoch + 1):
            model.train()
            h, a, zs, _, cs, qs = model(xs)
            loss, first_order, second_order, cross, cluster, entropy, uniform = trainer.getLoss(qs, a, xs, cs, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % args.print_every == 0 or epoch == 1:
                logging.info(
                    'Epoch %d/%d loss=%.4f first_order=%.4f second_order=%.4f cross=%.4f cluster=%.4f entropy=%.4f uniform=%.4f',
                    epoch, args.epoch, loss.item(), first_order.item(), second_order.item(), cross.item(),
                    cluster.item(), entropy.item(), uniform.item(),
                )
            if epoch % 100 == 0:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)

    model.eval()
    with torch.no_grad():
        h, a, zs, _, cs, qs = model(xs)
    save_outputs(output_dir, h, a, zs, cs, qs)


if __name__ == '__main__':
    main()
