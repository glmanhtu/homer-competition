import wandb
from options.cross_val_options import CrossValOptions
from train import Trainer

args = CrossValOptions().parse()

if __name__ == "__main__":
    assert args.n_epochs_per_eval == args.nepochs, "In cross-validation, n_epochs_per_eval have to be equal to nepochs"
    for fold in range(args.k_fold):
        run = wandb.init(group=args.group,
                         name=f'{args.name}_fold-{fold}',
                         project=args.wb_project,
                         entity=args.wb_entity,
                         resume=args.resume,
                         config=args,
                         mode=args.wb_mode)

        trainer = Trainer(fold=fold, k_fold=args.k_fold)
        if trainer.is_trained():
            trainer.load_pretrained_model()

        if args.resume or not trainer.is_trained():
            trainer.train()

        run.finish()
