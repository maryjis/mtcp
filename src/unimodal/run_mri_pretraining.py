# Standard libraries
import argparse
import os

# Third-party libraries
import torch
import wandb
import numpy as np
import pandas as pd

# Local libraries
from src.utils import seed_everything, seed_worker
from src.unimodal.mri.datasets import DatasetBraTSTumorCentered
from src.unimodal.mri.models import MRIEncoder
from src.unimodal.mri.transforms import get_tumor_transforms
from src.unimodal.commons.optim_contrastive import training_loop_contrastive
from src.unimodal.commons.losses import ContrastiveLoss
from src.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/BraTS_2023/MRI")
    parser.add_argument("--path_to_dataset_file", type=str, default="src/data/dataset.csv") #remember, counted from the place where the script was launched
    parser.add_argument("--path_to_save", type=str, default="outputs/models") #will be created if not exists
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--modalities", nargs="+", default=["t1ce", "flair"])
    parser.add_argument("--modalities", nargs="+", default=["t1c"]) #["t1c", "t2f"]
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--entity", type=str, default="dmitriykornilov_team") #define will be used wandb logging or not
    parser.add_argument("--project", type=str, default="cancer_mtcp")
    parser.add_argument("--run_name", type=str, default="mri_pretraining")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--tumor_centered", type=bool, default=True)
    # parser.add_argument("--n_cpus", type=int, default=40)
    # parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--n_cpus", type=int, default=8)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1999)
    parser.add_argument("--train_percentage", type=float, default=0.75) #test patients are not included in train/val
    parser.add_argument("--use_monai_weights", type=bool, default=False)
    parser.add_argument("--model_name_postfix", type=str, default="")
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)
    patients = os.listdir(args.data_path)
    logger.info("Total patients", len(patients))

    #get patients only with target modalities
    patients_with_needed_modalities = []
    needed_modalities = set(args.modalities)
    needed_modalities.add("seg") #segmentation mask used to compute center of tumor
    for patient in patients:
        available_modalities = set([x.split("-")[-1].split(".")[0] for x in os.listdir(os.path.join(args.data_path, patient))])
        if needed_modalities.intersection(available_modalities) == needed_modalities:
            patients_with_needed_modalities.append(patient)
    patients = patients_with_needed_modalities
    logger.info(f"Patients with all needed modalities: {len(patients)}")

    dataframe = pd.read_csv(args.path_to_dataset_file)
    dataframe_test = dataframe[dataframe["group"] == "test"]
    # get patient ids where MRI is not NaN
    dataframe_test = dataframe_test[~dataframe_test["MRI"].isna()]
    patients_to_exclude = [
        patient_path.split("/")[-1] for patient_path in dataframe_test.MRI.values
    ]
    patients = [patient for patient in patients if patient not in patients_to_exclude]
    logger.info("Included patients (pre-train)", len(patients))
    logger.info("Excluded patients (further test)", len(patients_to_exclude))

    # split patient into train and val by taking random 70% of patients for training
    train_patients = np.random.choice(
        patients, int(len(patients) * args.train_percentage), replace=False
    )
    val_patients = [patient for patient in patients if patient not in train_patients]
    logger.info("Performing contrastive training on BraTS dataset.")
    logger.info("Modalities used : {}.", args.modalities)
    if bool(args.tumor_centered):
        logger.info("Using tumor centered dataset.")
        category = "tumor"
        sizes = (64, 64, 64)
        train_dataset = DatasetBraTSTumorCentered(
            args.data_path,
            args.modalities,
            patients=train_patients,
            sizes=sizes,
            return_mask=False,
            transform=get_tumor_transforms(sizes), #used to generate 2 positive samples from 1
        )
        val_dataset = DatasetBraTSTumorCentered(
            args.data_path,
            args.modalities,
            patients=val_patients,
            sizes=sizes,
            return_mask=False,
            transform=get_tumor_transforms(sizes),
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.n_cpus,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpus,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )

    model = MRIEncoder(
        projection_head=True, 
        in_channels=len(args.modalities),
        use_monai_weights=args.use_monai_weights,
    )

    logger.info(
        "Number of parameters : {}",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.n_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpus)))

    model.to(device)
    logger.info("Using {} gpus to train the model", args.n_gpus)
    torch.cuda.reset_peak_memory_stats(device=device)
    logger.info(
        f"gpu used {torch.cuda.max_memory_allocated(device=device)} memory"
    )

    contrastive_loss = ContrastiveLoss(temperature=args.temperature, k=args.k)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=5e-6
    )

    path_to_save = (
        f"{args.path_to_save}/{'-'.join(args.modalities)}_tumor{str(args.tumor_centered)}{str(args.model_name_postfix)}.pth"
    )
    # if a wandb entity is provided, log the training on wandb
    wandb_logging = True if args.entity is not None else False
    if wandb_logging:
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=args.run_name,
            reinit=True,
            config=vars(args),
        )
    logger.info("Training started!")
    _, _ = training_loop_contrastive(
        model,
        args.epochs,
        contrastive_loss,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        device,
        path_to_save,
        wandb_logging=wandb_logging,
        k=args.k,
    )
    if wandb_logging:
        run.finish()
    logger.info("Training finished!")
