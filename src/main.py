from src import datasets, utils, cam, model, train
from datasets import *
from utils import *
from cam import *
from model import *
from train import *

if __name__ == '__main__':

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = datasets.get_datasets()
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")

    # Load the training and validation data loaders.
    train_loader, valid_loader = datasets.get_data_loaders(dataset_train, dataset_valid)

    # Learning_parameters.
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")

    # Load the model.
    model = build_model(
        pretrained=args['pretrained'],
        fine_tune=args['fine_tune'],
        num_classes=len(dataset_classes)
    ).to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function.
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        verbose=True
    )

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader,
            optimizer, criterion,
            scheduler=scheduler, epoch=epoch
        )
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                     criterion, dataset_classes)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)
        time.sleep(5)

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion)

    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')
