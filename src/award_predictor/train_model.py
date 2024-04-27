from test_model import evaluate

def epoch_string(epoch, epochs, train_loss, test_loss, f1):
    return ' | '.join([
        f"Epoch[{epoch}/{epochs}]:",
        f"Train Loss: {train_loss:>5f}",
        f"Test Loss: {test_loss:>5f}",
        f"F1-Score: {f1:>5f}"])

def train(train_dataloader, test_dataloader,
          model, loss_fn, optimizer, epochs, device,
          logger, out_dir):
    best_loss = 999999
    best_f1 = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):
            y = y.unsqueeze(1)
            X, y = X.to(device), y.to(device)

            # Compute prediction and loss
            _, logits = model(X)
            loss = loss_fn(logits, y)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = train_loss / (batch + 1)
        _, _, _, test_loss, acc, f1 = evaluate(
            model, loss_fn, test_dataloader, device)

        print("\r", end="")
        print(epoch_string(epoch, epochs, train_loss, test_loss, f1), end="")

        logger.store({"train_loss": train_loss, "test_loss": test_loss,
                      "accuracy": acc, "f1-score": f1})

        logger.save_csv(str(out_dir / "log.csv"))

        if test_loss < best_loss:
            best_loss = test_loss
            model.save(str(out_dir / "best_loss.pt"))
        if f1 > best_f1:
            best_f1 = f1
            model.save(str(out_dir / "best_f1.pt"))

    print()