train_cost_list = []
train_mis_rate_list = []
valid_cost_list = []
valid_mis_rate_list = []

for i in range(C_range.size):
    coef = np.mean(lr.coefs_paths_[1][:, i], axis=0)
    lr.coef_ = coef[:-1].reshape(1, -1)
    lr.intercept_ = coef[-1].reshape(1)
    train_y_hat = lr.predict_proba(train_X_scaled)
    cost = log_loss(train_y, train_y_hat)
    # cost = clf.cost(torch.tensor(train_y), torch.tensor(train_y_hat[:,1])).cpu()
    mis_rate = np.mean((train_y.flatten() != (train_y_hat[:, 1] > 0.5)))
    train_cost_list.append(cost)
    train_mis_rate_list.append(mis_rate)

    valid_X_scaled = scaler.transform(valid_X)
    valid_y_hat = lr.predict_proba(valid_X_scaled)
    cost = log_loss(valid_y, valid_y_hat)
    mis_rate = np.mean((valid_y.flatten() != (valid_y_hat[:, 1] > 0.5)))
    valid_cost_list.append(cost)
    valid_mis_rate_list.append(mis_rate)


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax1.plot(C_range, train_cost_list, label="train set")
ax1.plot(C_range, valid_cost_list, label="valid set")
ax1.set_xlabel("C")
ax1.set_xscale("log")
ax1.set_ylabel("Cost")
ax1.set_title("Cost vs C")
ax1.legend()
ax1.set_ylim(0, 0.3)
ax2.plot(C_range, train_mis_rate_list, label="train set")
ax2.plot(C_range, valid_mis_rate_list, label="valid set")
ax2.set_xlabel("C")
ax2.set_xscale("log")
ax2.set_ylabel("Misclassification rate")
ax2.set_title("Misclassification rate vs C")
ax2.legend()
ax2.set_ylim(0, 0.2)