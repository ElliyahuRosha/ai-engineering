def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_deriv, X_train, Y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(X_train, Y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_deriv(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(X_train)
        if verbose and e % 10 == 0:
            print(f"{e + 1}/{epochs}, error={error}")