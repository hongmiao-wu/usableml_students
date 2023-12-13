import threading
import queue
import webbrowser

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
from torch import manual_seed, Tensor
from torch.optim import Optimizer, SGD

from ml_utils.model import ConvolutionalNeuralNetwork
from ml_utils.training import training


app = Flask(__name__)
socketio = SocketIO(app)


# Initialize variables
seed = 42
acc = -1
loss = 1
lr = 0.3
epochs = 10
batch_size = 256
q_acc = queue.Queue()
q_loss = queue.Queue()



def listener():
    global q_acc, q_loss, acc, loss
    while True:
        acc = q_acc.get()
        loss = q_loss.get()
        q_acc.task_done()
        q_loss.task_done()


@app.route("/", methods=["GET", "POST"])
def index():
    global seed, acc, loss, lr, epochs, batch_size
    # render "index.html" as long as user is at "/"
    return render_template("index.html", seed=seed, acc=acc, loss=loss, lr=lr, epochs=epochs, batch_size=batch_size)

@app.route("/start_training", methods=["POST"])
def start_training():
    # ensure that these variables are the same as those outside this method
    global q_acc, q_loss, seed, lr, epochs, batch_size
    # determine pseudo-random number generation
    manual_seed(seed)
    np.random.seed(seed)
    # initialize training
    model = ConvolutionalNeuralNetwork()
    opt = SGD(model.parameters(), lr=lr, momentum=0.5)
    print(seed)
    print(lr)
    print(epochs)
    print(batch_size)
    # execute training
    training(model=model,
             optimizer=opt,
             cuda=False,
             n_epochs=epochs,
             batch_size=batch_size,
             q_acc=q_acc,
             q_loss=q_loss)
    return jsonify({"success": True})


@app.route("/update_seed", methods=["POST"])
def update_seed():
    global seed
    seed = int(request.form["seed"])
    return jsonify({"seed": seed})

#adjust learning rate 
@app.route("/update_learningRate", methods=["POST"])
def update_learningRate():
    global lr
    lr = float(request.form["lr"])
    return jsonify({"lr": lr})

#adjust number of epochs
@app.route("/update_numEpochs", methods=["POST"])
def update_numEpochs():
    global epochs
    epochs = int(request.form["epochs"])
    return jsonify({"epochs": epochs})

#adjust batch_size
@app.route("/update_batch_size", methods=["POST"])
def update_batch_size():
    global batch_size
    batch_size = int(request.form["batch_size"])
    return jsonify({"batch_size": batch_size})

@app.route("/get_accuracy")
def get_accuracy():
    global acc
    return jsonify({"acc": acc})

@app.route("/get_loss")
def get_loss():
    global loss
    return jsonify({"loss": loss})

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5001
    print("App started")
    threading.Thread(target=listener, daemon=True).start()
    webbrowser.open_new_tab(f"http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=True)
