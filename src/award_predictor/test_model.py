import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from torch import no_grad, round

import csv

def evaluate(model, loss_fn, dataloader, device):
    model.eval()
    with no_grad():
        X_test, y_test = dataloader.dataset.X, dataloader.dataset.y.unsqueeze(1)
        X_test, y_test = X_test.to(device), y_test.to(device)
        y_prob, logits = model(X_test)
        loss = loss_fn(logits, y_test).item()

        y_test = y_test.detach().cpu().numpy()
        y_pred = round(y_prob).detach().cpu().numpy()
        y_prob = y_prob.detach().cpu().numpy()

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        return y_test, y_pred, y_prob, loss, acc, f1
    
def save_plot(save_dir, f_name):
    plt.savefig(f'{save_dir}/{f_name}.pdf', bbox_inches='tight', dpi=600)
    plt.savefig(f'{save_dir}/{f_name}.png', bbox_inches='tight', dpi=600)

def plot_confusion_matrix(y_test, y_pred, save_dir):
    fig, ax = plt.subplots(figsize=(3, 3))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax,
                                            cmap='gray', colorbar=False)
    # plt.title('Confusion Matrix')

    save_plot(save_dir, 'confusion_matrix')

def plot_roc_curve(y_test, y_prob, save_dir):
    fig = plt.figure(figsize=(3, 3))

    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             linestyle='-', marker='s', label='Logistic')
    plt.grid()

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    plt.legend(fontsize=6, frameon=False)

    plt.text(0.8, 0.1, f'AUC: {auc_score:.3f}',
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})

    save_plot(save_dir, 'roc_curve')

    return auc_score

def plot_precision_recall_curve(y_test, y_prob, save_dir):
    fig = plt.figure(figsize=(3, 3))

    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill],
             linestyle='--', label='No Skill')

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_score = auc(recall, precision)

    plt.plot(recall, precision,
             linestyle='-', marker='s', label='Logistic')
    plt.grid()

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision Recall Curve')
    plt.legend(fontsize=6, frameon=False)

    plt.text(0.1, 0.2, f'AUC: {auc_score:.3f}',
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})

    save_plot(save_dir, 'precision_recall_curve')

    return auc_score

def save_csv(save_dir, dict):
    csv_path = save_dir / "results.csv"
    with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(dict.keys()) # header
            writer.writerow(dict.values()) # data


def test_and_plot(out_dir, model, loss_fn, test_dataloader, device):

    y_test, y_pred, y_prob, loss, acc, f1 = evaluate(
        model, loss_fn, test_dataloader, device)

    plot_confusion_matrix(y_test, y_pred, out_dir)
    roc_auc = plot_roc_curve(y_test, y_prob, out_dir)
    pr_auc = plot_precision_recall_curve(y_test, y_prob, out_dir)

    # print results on console
    print("############ RESULTS ############") 
    print("Loss:", loss)
    print("Average accuracy:", acc)
    print("f1-score (macro):", f1)
    print(f"roc auc: {roc_auc}")
    print(f"precision-recall auc: {pr_auc}")
    print("#################################")

    # save results on csv
    save_csv(out_dir, {
        "loss": loss,
        "accuracy": acc,
        "f1-score": f1,
        "roc-auc": roc_auc,
        "pr-auc": pr_auc
    })
    
