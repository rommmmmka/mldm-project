import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm.notebook import tqdm

from lightsleepnet import LightSleepNet


SLEEP_STAGES = ["W", "N1", "N2", "N3", "REM"]

def train_test_loop(
    data: list,
    labels: list,
    test_subject: int,
    logging: int | None = None
):
    subjects_data_train = data[:test_subject - 1] + data[test_subject:]
    subjects_label_train = labels[:test_subject - 1] + labels[test_subject:]
    subjects_data_train_tensor = torch.cat(subjects_data_train)
    subjects_label_train_tensor = torch.cat(subjects_label_train)
    train_dataset = torch.utils.data.TensorDataset(subjects_data_train_tensor, subjects_label_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    subject_data_test_tensor = data[test_subject - 1]
    subject_label_test_tensor = labels[test_subject - 1]
    test_dataset = torch.utils.data.TensorDataset(subject_data_test_tensor, subject_label_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LightSleepNet().to("cuda")
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for training_epoch in tqdm(range(100), desc="Training"):
        train_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch.requires_grad = True

            outputs = model(X_batch).float()
            loss = criterion(outputs, y_batch)

            weights = []

            for epoch_idx in range(X_batch.size(0)):
                epoch_data = X_batch[epoch_idx].unsqueeze(0)
                epoch_loss = criterion(model(epoch_data), y_batch[epoch_idx].unsqueeze(0))

                epoch_grads = torch.autograd.grad(epoch_loss, epoch_data, retain_graph=True)[0]
                grad_norms = epoch_grads.norm(p=2, dim=1).squeeze(0)

                delta = 0.1 * grad_norms.std().item()
                density = ((grad_norms.unsqueeze(1) - grad_norms.unsqueeze(0)).abs() < delta).sum(dim=1).float()

                epoch_weight = density.mean()
                weights.append(epoch_weight.detach())

            weights = torch.tensor(weights, device="cuda")
            weights /= weights.sum()
            # print(weights)
            weighted_loss = (weights * loss).sum()
            train_loss += weighted_loss.item()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)

        if (
            logging is not None and (
                training_epoch == 0 or
                (training_epoch + 1) % logging == 0
            )
        ):
            print(f"{training_epoch + 1:<5}Loss: {train_loss / len(train_loader):.3f}  Accuracy: {100 * correct / total:.3f}")

    y_pred = []
    test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            y_pred.append(predicted.cpu())

    y_pred = torch.cat(y_pred).numpy()
    y_true = subject_label_test_tensor.to("cpu").numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    print("TEST RESULTS")
    print(f"Loss: {test_loss / len(test_loader):.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    for sleep_stage, precision_val, recall_val, f1_val in zip(SLEEP_STAGES, precision, recall, f1):
        print(f"{sleep_stage:7}precision={precision_val:.3f}  recall={recall_val:.3f}  f1={f1_val:.3f}")
    print(confusion_matrix(y_true, y_pred))
