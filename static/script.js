var slider1 = document.getElementById("slider1"); //seed slider
var slider2 = document.getElementById("slider2"); //learing rate slider
var sliderValueElement1 = document.getElementById("sliderValue1");
var sliderValueElement2 = document.getElementById("sliderValue2");
var playButton = document.getElementById("playButton");
var accuracyElement = document.getElementById("accuracy");
var lossElement = document.getElementById("loss");

// Function to disable slider and button when button is pressed
playButton.addEventListener('click', function () {
    startTraining();
    slider1.disabled = true;
    slider2.disabled = true;
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
function updateSliderValue() {
    sliderValueElement1.textContent = slider1.value;
}

// Function to update slider value display for learning rate
function updateSliderValue() {
    sliderValueElement2.textContent = slider2.value;
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
slider1.addEventListener("input", function() {
    updateSliderValue();
    fetch("/update_seed", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: "seed=" + slider1.value
    });
});

// Function to change  learning rate when slider is changed
slider2.addEventListener("input", function() {
    updateSliderValue();
    fetch("/update_learningRate", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: "learing rate=" + slider2.value
    });
});