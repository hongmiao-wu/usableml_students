var slider_seed = document.getElementById("slider_seed"); //seed slider
var slider_lr = document.getElementById("slider_lr"); //learing rate slider
var slider_ep = document.getElementById("slider_ep"); //num epochs slider
var slider_batches = document.getElementById("slider_batches"); //num epochs slider
var sliderValueElement_seed = document.getElementById("sliderValue_seed");
var sliderValueElement_lr = document.getElementById("sliderValue_lr");
var sliderValueElement_ep = document.getElementById("sliderValue_ep");
var sliderValueElement_batches = document.getElementById("sliderValue_batches");
var playButton = document.getElementById("playButton");
var accuracyElement = document.getElementById("accuracy");
var lossElement = document.getElementById("loss");

// Function to disable slider and button when button is pressed
playButton.addEventListener('click', function () {
    startTraining();
    slider_seed.disabled = true;
    slider_lr.disabled = true;
    slider_ep.disabled = true;
    slider_batches.disabled = true;
    playButton.disabled = true;
});

// stopButton.addEventListener('click', function(){

// }

// Function to update accuracy value on the page
function updateAccuracy() {
    fetch("/get_accuracy")
        .then(response => response.json())
        .then(data => {
            accuracyElement.textContent = data.acc;
        });
}

function updateLoss() {
    fetch("/get_loss")
        .then(response => response.json())
        .then(data => {
            lossElement.textContent = data.loss;
        });
}

// Function to update slider value display for seed
function updateSliderValue_seed() {
    sliderValueElement_seed.textContent = slider_seed.value;
}

// Function to update slider value display for learning rate
function updateSliderValue_lr() {
    sliderValueElement_lr.textContent = slider_lr.value;
}

// Function to update slider value display for number of epochs
function updateSliderValue_ep() {
    sliderValueElement_ep.textContent = slider_ep.value;
}

// Function to update slider value display for batch size
function updateSliderValue_batches() {
    sliderValueElement_batches.textContent = slider_batches.value;
}

// Function to start training
function startTraining() {
    fetch("/start_training", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        }
    })
    .then(response => response.json())
}

// Update accuracy every second
setInterval(function() {
    updateAccuracy();
    updateLoss();
}, 1000);

// Function to change seed value when slider is changed
slider_seed.addEventListener("input", function() {
    updateSliderValue_seed();
    fetch("/update_seed", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: "seed=" + slider_seed.value
    });
});

// Function to change learning rate when slider is changed
slider_lr.addEventListener("input", function() {
    updateSliderValue_lr();
    fetch("/update_learningRate", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: "lr=" + slider_lr.value
    });
});

// Function to change number of epochs when slider is changed
slider_ep.addEventListener("input", function() {
    updateSliderValue_ep();
    fetch("/update_numEpochs", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: "epochs=" + slider_ep.value
    });
});

// Function to change size of batches when slider is changed
slider_batches.addEventListener("input", function() {
    updateSliderValue_batches();
    fetch("/update_batch_size", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: "batch_size=" + slider_batches.value
    });
});batch_size