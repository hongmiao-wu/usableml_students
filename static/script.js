var slider = document.getElementById("slider");
var sliderValueElement = document.getElementById("sliderValue");
var playButton = document.getElementById("playButton");
var stopButton = document.getElementById("stopButton");
var resumeButton = document.getElementById("resumeButton");
var accuracyElement = document.getElementById("accuracy");
var lossElement = document.getElementById("loss");
var epochElement = document.getElementById("epoch");
var epochLossesElement = document.getElementById("epoch_losses");
var imageElement = document.getElementById("loss_plot");

// Function to disable slider and button when button is pressed
playButton.addEventListener('click', function () {
    startTraining();
    updateImage();
    slider.disabled = true;
    playButton.disabled = true;
});

stopButton.addEventListener('click', function(){
    stopTraining();
    // saveCheckpoint();
    slider.disabled = false;
    stopButton.disabled = true;
    playButton.disabled = false;
});

resumeButton.addEventListener('click', function(){
    resumeTraining();
    slider.disabled = true;
    stopButton.disabled = false;
    resumeButton.disabled = true;
    playButton.disabled = false;
});

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

function updateEpoch() {
    fetch("/get_epoch")
        .then(response => response.json())
        .then(data => {
            epochElement.textContent = data.epoch;
        });
}

function updateEpochLosses() {
    fetch("/get_epoch_losses")
        .then(response => response.json())
        .then(data => {
            epochLossesElement.textContent = data.epoch_losses;
        });
}

function updateImage() {
    fetch("/get_loss_image")  
        .then(response => response.json())  
        .then(data => {
            imageElement = data.loss_img_url;
        });
}


// Function to update slider value display
function updateSliderValue() {
    sliderValueElement.textContent = slider.value;
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

function stopTraining(){
    fetch("/stop_training", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        }
    })
    .then(response => response.json())
}

// function saveCheckpoint() {
//     fetch("/save_checkpoint", {
//         method: "POST",
//         headers: {
//             "Content-Type": "application/x-www-form-urlencoded"
//         }
//     })
//     .then(response => response.json());
// }

function resumeTraining(){
    fetch("/resume_training", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        }
    })
    .then(response => response.json())
}

// Update every second
setInterval(function() {
    updateAccuracy();
    updateLoss();
    updateEpoch();
    updateEpochLosses();
    updateImage();
}, 1000);



// Function to change seed value when slider is changed
slider.addEventListener("input", function() {
    updateSliderValue();
    fetch("/update_seed", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: "seed=" + slider.value
    });
});
