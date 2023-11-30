import threading
import queue
import webbrowser
import base64

from io import BytesIO
from matplotlib.figure import Figure

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
from torch import manual_seed, Tensor
from torch.optim import Optimizer, SGD
import matplotlib.pyplot as plt

from ml_utils.model import ConvolutionalNeuralNetwork
from ml_utils.training import training


app = Flask(__name__)
socketio = SocketIO(app)


# Initialize variables
seed = 42
acc = -1
loss = 0.1
epoch_losses = []
stop_signal = False
q_acc = queue.Queue()
q_loss = queue.Queue()
q_stop_signal = queue.Queue()



def listener():
    global q_acc, q_loss, q_stop_signal, acc, loss, stop_signal
    while True:
        acc = q_acc.get()
        loss = q_loss.get()
        q_stop_signal.put(stop_signal)
        q_acc.task_done()
        q_loss.task_done()
        q_stop_signal.task_done()
        


@app.route("/", methods=["GET", "POST"])
def index():
    global seed, acc, loss, epoch_losses

    # render "index.html" as long as user is at "/"
    # return render_template("index.html", seed=seed, acc=acc, loss=loss)

    
    """
    If you want to show the plot directly on the index page
    """
    epoch_losses.append(loss)
    
    fig = Figure()
    ax = fig.subplots()  # Create a new figure with a single subplot
    ax.plot(epoch_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_title('Training Loss per Epoch')
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    
    dataurl = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    return render_template("index.html", seed=seed, acc=acc, loss=loss, loss_plot=dataurl)
    

@app.route("/start_training", methods=["POST"])
def start_training():
    # ensure that these variables are the same as those outside this method
    global q_acc, q_loss, seed, stop_signal
    
    # determine pseudo-random number generation
    manual_seed(seed)
    np.random.seed(seed)
    # initialize training
    model = ConvolutionalNeuralNetwork()
    opt = SGD(model.parameters(), lr=0.3, momentum=0.5)
    # execute training
    training(model=model,
             optimizer=opt,
             cuda=False,
             n_epochs=10,
             start_epoch=0,
             batch_size=256,
             q_acc=q_acc,
             q_loss=q_loss,
             q_stop_signal=q_stop_signal)
    return jsonify({"success": True})

@app.route("/stop_training", methods=["POST"])
def stop_training():
    global stop_signal
    stop_signal = True  # Set the stop signal to True
    # saveCheckpoint()
    return jsonify({"success": True})

# @app.route("/resume_training", methods=["POST"])
# def resume_training():
#     global stop_signal
#     PATH = "stop.pt"
#     stop_signal = False  # Set the stop signal to False
#     model = ConvolutionalNeuralNetwork()
#     opt = SGD(model.parameters(), lr=0.3, momentum=0.5)
#     checkpoint = torch.load(PATH)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     training(model=model,
#              optimizer=opt,
#              cuda=False,
#              n_epochs=10,
#              start_epoch=epoch,
#              batch_size=256,
#              q_acc=q_acc,
#              q_loss=q_loss,
#              q_stop_signal=q_stop_signal)
#     return jsonify({"success": True})

@app.route("/loss_plot", methods=["GET"])
def loss_plot():
    global epoch_losses, loss

    epoch_losses.append(loss)
    fig = Figure()
    ax = fig.subplots()  # Create a new figure with a single subplot
    ax.plot(epoch_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_title('Training Loss per Epoch')
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

@app.route("/update_seed", methods=["POST"])
def update_seed():
    global seed
    seed = int(request.form["seed"])
    return jsonify({"seed": seed})


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
